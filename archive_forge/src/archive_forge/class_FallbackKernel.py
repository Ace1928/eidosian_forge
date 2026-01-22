import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
class FallbackKernel(ExternKernelAlloc):
    args_default_value: List[Dict[str, Any]]

    def __init__(self, layout, kernel, tensor_args, nontensor_args, unflatten_args, kwargs=None):
        super().__init__(layout, tuple(tensor_args), tuple(nontensor_args))
        self.outputs: Sequence[Any] = []
        self.use_runtime_dispatch = False
        self.abi_compatible_kernel = None
        assert isinstance(kernel, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)), f'Fails to create FallbackKernel for {kernel}: {type(kernel)} not supported'
        self.op_overload = kernel
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        V.graph.warn_fallback(self.kernel)

    def set_cpp_kernel(self, kernel):
        from .codegen.wrapper import get_cpp_op_schema
        assert not kernel._schema.is_mutable, f'mutable {kernel.__name__} is not supported with cpp_wrapper'

        def is_not_write(arg):
            return arg.alias_info is None or not arg.alias_info.is_write
        assert all((is_not_write(x) for x in kernel._schema.arguments)), f'{kernel.__name__} with alias_info arguments is not supported with cpp_wrapper'
        assert all((is_not_write(x) for x in kernel._schema.returns)), f'{kernel.__name__} with alias_info returns is not supported with cpp_wrapper'
        self.cpp_kernel = kernel._schema.name
        self.cpp_kernel_overload_name = kernel._schema.overload_name
        self.cpp_kernel_key = f'{self.cpp_kernel.replace('::', '_')}_{self.cpp_kernel_overload_name}'
        self.cpp_op_schema = get_cpp_op_schema(kernel)
        self.ordered_kwargs_for_cpp_kernel = [x.name for x in kernel._schema.arguments if x.kwarg_only]

    def is_legacy_abi_kernel(self):
        return '_scaled_dot_product_flash_attention' in str(self.kernel)

    def get_arg_default_value(self, pos):
        assert hasattr(self, 'args_default_value'), 'self.args_default_value has to be provided'
        assert pos < len(self.args_default_value), f'expected the index {pos} to be smaller than len(self.args_default_value): {len(self.args_default_value)}'
        return self.args_default_value[pos]['value']

    def _get_abi_compatible_kernel(self):
        if not V.graph.cpp_wrapper:
            return self.kernel

        def sdpa_ver_fn():
            if any((self.get_kwargs_value(arg_name) is None for arg_name in self.ordered_kwargs_for_cpp_kernel)):
                return f'{self.cpp_kernel}_v2'
            else:
                return self.cpp_kernel
        kernel_to_ver = {'at::_scaled_dot_product_flash_attention': sdpa_ver_fn}
        if (ver_fn := kernel_to_ver.get(self.cpp_kernel, None)) is not None:
            return ver_fn()
        return self.cpp_kernel

    def codegen_args(self):

        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self):
                return self.ref
        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        self.abi_compatible_kernel = self._get_abi_compatible_kernel()
        if V.graph.cpp_wrapper and isinstance(self.op_overload, torch._ops.OpOverload):
            args = [V.graph.wrapper_code.val_to_cpp_arg_str(param.real_type, x, self.is_legacy_abi_kernel()) for param, x in zip(self.op_overload._schema.arguments, args)]
        else:
            args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]
        if V.graph.cpp_wrapper and hasattr(self, 'args_default_value'):
            n_args = len(args)
            n_pos_args = len(self.args_default_value)
            if n_args < n_pos_args:
                pos_args = [self.get_arg_default_value(i) for i in range(n_args, n_pos_args)]
                pos_args = [V.graph.wrapper_code.val_to_arg_str(x) for x in pos_args]
                args.extend(pos_args)
        self.kwargs.update(kwargs)
        return args

    @staticmethod
    def find_device(tensor_args, example_output):
        if tensor_args:
            return tensor_args[0].get_device()
        if isinstance(example_output, torch.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            devices = {FallbackKernel.find_device(None, x) for x in example_output}
            devices = [device for device in devices if device]
            if len(devices) == 1:
                return devices[0]
            for device in devices:
                if device.type == 'cuda':
                    return device
            return devices[0]
        return None

    def has_side_effects(self):
        if not isinstance(self.op_overload, torch._ops.OpOverload):
            return False
        return get_schema_info(self.op_overload).is_mutable()

    def get_alias_names(self):
        if not isinstance(self.op_overload, torch._ops.OpOverload):
            return []
        if torch._inductor.utils.is_view(self.op_overload):
            return [inp.get_name() for inp in self.inputs]
        return []

    def export_extern_kernel_node(self):
        assert isinstance(self, FallbackKernel)
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        ordered_kwargs = [kwargs.get(key, None) for key in self.ordered_kwargs_for_cpp_kernel]
        serializer = GraphModuleSerializer(None, None)
        named_arguments = serializer.serialize_inputs(self.op_overload, args, kwargs)

        def handle_single_output(return_type, output):
            if isinstance(return_type, torch.TensorType):
                out = output
                if isinstance(output, (list, tuple)):
                    assert len(output) == 1
                    out = output[0]
                return export_schema.Argument.create(as_tensor=export_schema.TensorArgument(name=out.get_name()))
            elif isinstance(return_type, torch.ListType) and isinstance(return_type.getElementType(), torch.TensorType):
                return export_schema.Argument.create(as_tensors=[export_schema.TensorArgument(name=out.get_name()) for out in output])
            else:
                raise RuntimeError(f'Unsupported return type {type(return_type)}')
        target = self.op_overload
        returns = target._schema.returns
        if len(returns) == 1:
            return_type = returns[0].real_type
            output_arguments = [handle_single_output(return_type, self.outputs)]
        else:
            assert isinstance(self.outputs, tuple)
            assert len(returns) == len(self.outputs)
            output_arguments = [handle_single_output(return_schema.real_type, output) for return_schema, output in zip(returns, self.outputs)]
        node = ExternKernelNode(name=self.get_name(), node=export_schema.Node(target=self.op_overload.name(), inputs=named_arguments, outputs=output_arguments, metadata={}))
        V.graph.extern_kernel_nodes.append(node)
        return [*args, *ordered_kwargs]

    def codegen(self, wrapper):
        kernel = self.op_overload
        if kernel.namespace == 'aten':
            assert isinstance(kernel, torch._ops.OpOverload)
            op_base_name = kernel.__name__.split('.')[0]
            if V.graph.cpp_wrapper:
                if config.is_fbcode() and kernel not in has_c_shim:
                    log.warning('%s is missing a c-shim implementation, using proxy executor as fallback', kernel)
                    self.use_runtime_dispatch = True
                    self.set_cpp_kernel(kernel)
                else:
                    self.cpp_kernel = f'at::{op_base_name}' if kernel._overloadname == 'default' else f'at::_ops::{kernel.__name__.replace('.', '_')}::call'
                    schema = kernel._schema
                    self.args_default_value = [{'type': x.real_type, 'value': x.default_value} for x in schema.arguments if not x.kwarg_only]
                    self.ordered_kwargs_for_cpp_kernel = [x.name for x in schema.arguments if x.kwarg_only]
                    self.kwargs_default_value = {x.name: {'type': x.real_type, 'value': x.default_value} for x in schema.arguments if x.kwarg_only}
            else:
                self.kernel = f'aten.{op_base_name}'
        elif isinstance(kernel, torch._ops.HigherOrderOperator):
            if getattr(torch._prims.rng_prims, kernel.__name__, None) is kernel:
                self.kernel = f'torch._prims.rng_prims.{kernel.__name__}'
            else:
                raise NotImplementedError('Unable to find HigherOrderOperator kernel name')
        elif V.graph.cpp_wrapper:
            self.use_runtime_dispatch = True
            self.set_cpp_kernel(kernel)
        else:
            self.kernel = f'{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}'
        if self.use_runtime_dispatch:
            self.codegen_comment(wrapper)
            exported_args = None
            args = None
            if config.is_fbcode() and V.graph.cpp_wrapper:
                exported_args = self.export_extern_kernel_node()
            else:
                args = [*self.codegen_args(), *self.codegen_kwargs()]
            wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.codegen_kernel_name(), args, self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overload_name, self.op_overload, exported_args, self.outputs)
        else:
            self.codegen_comment(wrapper)
            args = [*self.codegen_args(), *self.codegen_kwargs()]
            V.graph.wrapper_code.generate_fallback_kernel(self, args)
            if isinstance(self.layout, Layout):
                self.codegen_size_asserts(wrapper)

    @staticmethod
    def tensor_to_layout(output: torch.Tensor):
        return FixedLayout(output.device, output.dtype, convert_shape_to_inductor(output.size()), convert_shape_to_inductor(output.stride()))

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
        context = V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()
        with context:
            example_output, tensor_args, non_tensor_args, unflatten_args = cls.process_kernel(kernel, *args, **kwargs)
        device = cls.find_device(tensor_args, example_output)
        assert device, 'Not sure where to find device info'
        packed = cls(MultiOutputLayout(device), kernel, tensor_args, non_tensor_args, unflatten_args)

        def generate_output(output, indices):
            if isinstance(output, (list, tuple)):
                return type(output)((generate_output(output[i], indices + [(type(output), i)]) for i in range(len(output))))
            elif isinstance(output, dict):
                return {key: generate_output(val, indices + [(type(output), key)]) for key, val in output.items()}
            elif isinstance(output, torch.Tensor):
                return MultiOutput(cls.tensor_to_layout(output), packed, indices)
            elif isinstance(output, int):
                return output
            elif isinstance(output, torch.SymInt):
                return output.node.expr
            else:
                assert output is None, f'FallbackKernel output type {type(output)} is not supported'
                return None
        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple, dict)):
            packed.outputs = outputs
        else:
            packed.outputs = [outputs]
        return outputs

    def apply_constraint(self):
        return super().apply_constraint()