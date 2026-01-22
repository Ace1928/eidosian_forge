import inspect
import logging
import math
import re
import types
from typing import Dict, List
from torch._streambase import _StreamBase
from ..guards import install_guard
import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators
from .. import config, polyfill, variables
from ..allowed_functions import torch_get_name
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..utils import (
from .base import VariableTracker
from .ctx_manager import (
from .distributed import is_constant_pg_functions, is_from_local, ProcessGroupVariable
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lists import ListVariable, TupleVariable
from .torch_function import can_dispatch_torch_function, dispatch_torch_function
class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""

    def __repr__(self):
        return f'TorchInGraphFunctionVariable({self.value})'

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable, DeterministicAlgorithmsVariable, DisabledSavedTensorsHooksVariable, GradModeVariable, StreamContextVariable, SymNodeVariable, TensorVariable, UserDefinedObjectVariable
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls
        constant_args = check_constant_args(args, kwargs)
        unspec_python_args = check_unspec_python_args(args, kwargs)
        if self.can_constant_fold_through() and (constant_args or unspec_python_args):
            return ConstantVariable.create(self.as_python_constant()(*[x.as_python_constant() for x in args], **{k: v.as_python_constant() for k, v in kwargs.items()}))
        elif self.value in tracing_state_functions:
            assert not args and (not kwargs)
            if self.value in [torch._utils.is_compiling, torch._dynamo.external_utils.is_compiling]:
                tx.mark_inconsistent_side_effects()
            return ConstantVariable.create(tracing_state_functions[self.value])
        elif self.value in (torch._functorch.vmap.vmap_impl, torch._functorch.eager_transforms.grad_impl):
            return TorchHigherOrderOperatorVariable.make(self.value, source=self.source).call_function(tx, args, kwargs)
        elif self.value is torch.overrides.get_default_nowrap_functions:
            from .builder import SourcelessBuilder
            return SourcelessBuilder()(tx, torch.overrides.get_default_nowrap_functions())
        elif self.value == math.radians and (not (constant_args or unspec_python_args)):
            from .builder import SourcelessBuilder
            return tx.inline_user_function_return(SourcelessBuilder()(tx, polyfill.radians), args, kwargs)
        elif self.value in (torch.is_tensor, torch.overrides.is_tensor_like):
            assert len(args) == 1
            if isinstance(args[0], TensorVariable) or (self.value is torch.overrides.is_tensor_like and isinstance(args[0], UserDefinedObjectVariable) and hasattr(args[0].value, '__torch_function__')):
                return ConstantVariable.create(True)
            else:
                return ConstantVariable.create(False)
        elif self.value in (torch.is_floating_point, torch.is_complex):
            input_arg = None
            if args:
                input_arg = args[0]
            else:
                assert 'input' in kwargs
                input_arg = kwargs['input']
            if isinstance(input_arg, TensorVariable) and input_arg.dtype is not None:
                if self.value is torch.is_floating_point:
                    return ConstantVariable.create(input_arg.dtype.is_floating_point)
                elif self.value is torch.is_complex:
                    return ConstantVariable.create(input_arg.dtype.is_complex)
                else:
                    raise AssertionError(f'calling {self.value}')
        elif self.value is torch.numel and isinstance(args[0], TensorVariable) and (args[0].size is not None):
            return ConstantVariable.create(product(args[0].size))
        elif self.value in REWRITE_OPS_TO_TENSOR_SIZE_METHOD:
            assert len(args) == 1
            assert isinstance(args[0], TensorVariable)
            return args[0].call_method(tx, 'size', [], {})
        elif self.value in (torch.nn.modules.utils._single, torch.nn.modules.utils._pair, torch.nn.modules.utils._triple, torch.nn.modules.utils._quadruple, torch.nn.modules.utils._ntuple):
            return self._call_ntuple(tx, args, kwargs)
        elif self.value is torch.is_grad_enabled:
            assert not (args or kwargs)
            install_guard(GradModeVariable._guards_singleton)
            return ConstantVariable.create(torch.is_grad_enabled())
        elif self.value is torch.use_deterministic_algorithms and len(args) == 1:
            return DeterministicAlgorithmsVariable.create(tx, args[0].as_python_constant())
        elif self.value is torch.are_deterministic_algorithms_enabled:
            assert not (args or kwargs)
            install_guard(DeterministicAlgorithmsVariable._guards_singleton)
            return ConstantVariable.create(torch.are_deterministic_algorithms_enabled())
        elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
            assert len(args) == 1
            return DisabledSavedTensorsHooksVariable.create(tx, args[0].as_python_constant())
        elif self.value is torch._C._is_torch_function_enabled:
            assert not (args or kwargs)
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            return ConstantVariable.create(tx.output.torch_function_enabled)
        elif self.value in (torch.overrides.has_torch_function, torch.overrides.has_torch_function_variadic, torch.overrides.has_torch_function_unary):
            assert not kwargs
            return ConstantVariable.create(any((has_torch_function(a) for a in args)))
        elif any((self.value is method for method in [device_interface.stream for _, device_interface in get_registered_device_interfaces()])):
            assert len(args) == 1
            return StreamContextVariable.create(tx, args[0])
        elif self.value is torch.from_numpy:
            if not config.trace_numpy:
                unimplemented('torch.from_numpy. config.trace_numpy is False')
            if not np:
                unimplemented('torch.from_numpy. NumPy is not available')
            return wrap_fx_proxy_cls(target_cls=TensorVariable, tx=tx, proxy=tx.output.create_proxy('call_function', torch.as_tensor, *proxy_args_kwargs(args, {})), example_value=None)
        elif can_dispatch_torch_function(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)
        elif self.value is torch.jit.annotate:
            assert len(args) == 2
            return args[1]
        elif self.value is torch.backends.cudnn.is_acceptable:
            assert len(args) == 1 or 'tensor' in kwargs, 'Expect 1 input to cudnn.is_acceptable'
            tensor_variable = args[0] if len(args) > 0 else kwargs['tensor']
            assert isinstance(tensor_variable, TensorVariable), 'Expect input to cudnn.is_acceptable to be a tensor'
            tensor_inp = torch.tensor(0, dtype=tensor_variable.dtype, device=tensor_variable.device)
            return ConstantVariable.create(torch.backends.cudnn.is_acceptable(tensor_inp))
        elif self.value == torch.numel and len(args) == 1 and isinstance(args[0], TensorVariable) and (len(kwargs) == 0):
            return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_method', 'numel', *proxy_args_kwargs(args, kwargs)))
        elif self.value in [torch.ops.aten.sym_size, torch.ops.aten.sym_size.int] and len(args) == 2 and (len(kwargs) == 0) and isinstance(args[0], TensorVariable):
            return args[0].call_method(tx, 'size', [args[1]], {})
        elif self.value is [torch.ops.aten.sym_stride, torch.ops.aten.sym_stride.int] and len(args) == 2 and (len(kwargs) == 0) and isinstance(args[0], TensorVariable):
            return args[0].call_method(tx, 'stride', [args[1]], {})
        elif self.value == torch.addcdiv and len(args) == 3 and ('value' in kwargs) and (len(kwargs) == 1):
            result = TorchInGraphFunctionVariable(torch.div).call_function(tx, args[1:], {})
            result = TorchInGraphFunctionVariable(torch.mul).call_function(tx, [result, kwargs['value']], {})
            return TorchInGraphFunctionVariable(torch.add).call_function(tx, [args[0], result], {})
        elif is_constant_pg_functions(self.value):
            assert len(args) == 1, 'Expected one arg (pg)'
            assert isinstance(args[0], ProcessGroupVariable)
            invocation_result = self.value(args[0].as_python_constant())
            from .builder import SourcelessBuilder
            return SourcelessBuilder()(tx, invocation_result)
        elif is_from_local(self.value):
            args_as_value = [x.as_python_constant() for x in args[1:]]
            kwargs_as_value = {k: v.as_python_constant() for k, v in kwargs.items()}

            def fn_with_prim_types(x):
                return self.value(x, *args_as_value, **kwargs_as_value)
            fn_with_prim_types.__name__ = 'prim ' + self.value.__name__
            return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', fn_with_prim_types, *proxy_args_kwargs([args[0]], {})))
        elif self.value is torch.nested.nested_tensor and kwargs.get('layout', torch.strided) == torch.strided:
            raise unimplemented('torch.compile does not support strided NestedTensor')
        else:
            any_symints_or_symfloats = any((isinstance(x, SymNodeVariable) for x in args))
            all_ints_or_floats = all((isinstance(x, (variables.ConstantVariable, variables.SymNodeVariable)) for x in args))
            bin_ops = {'add', 'sub', 'mul', 'div', 'sqrt'}
            if getattr(self.value, '__module__', '') == 'torch' and self.value.__name__ in bin_ops and any_symints_or_symfloats and all_ints_or_floats:
                msg = f'Calling {str(self.value)} on only torch.SymInt arguments is not yet supported.\nTo support this behavior, we need to allow const-propping tensors that store symint data.\nFor now, dynamo will explicitly graph break when it encounters user code with this behavior.\n'
                log.warning(msg)
                raise unimplemented(msg)
            fn_ = self.value
            if any((isinstance(x, SymNodeVariable) for x in args)):
                if self.value == math.sqrt:
                    from torch.fx.experimental.sym_node import sym_sqrt
                    fn_ = sym_sqrt
            if fn_ is torch.tensor:

                def check_any_unspec(x):
                    if isinstance(x, (TensorVariable, SymNodeVariable)):
                        return True
                    elif isinstance(x, ListVariable):
                        return any((check_any_unspec(y) for y in x.items))
                    else:
                        return False
                data_arg = None
                if args:
                    data_arg = args[0]
                elif 'data' in kwargs:
                    data_arg = kwargs['data']
                if not isinstance(data_arg, TensorVariable) and check_any_unspec(data_arg):
                    fn_ = torch._refs.tensor
            tensor_variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', fn_, *proxy_args_kwargs(args, kwargs)))
            if isinstance(tensor_variable, TensorVariable) and 'requires_grad' in kwargs and kwargs['requires_grad'].as_python_constant():
                unimplemented('factory functions that return tensors that require grad are not supported.\nEither create the tensor outside the compiled region, or do not set the tensor to require_grad')
            if 'out' in kwargs and (not (isinstance(kwargs['out'], variables.ConstantVariable) and kwargs['out'].as_python_constant() is None)):
                if isinstance(tensor_variable, TupleVariable):
                    assert isinstance(kwargs['out'], (TupleVariable, ListVariable))
                    output_tensor_names = [tx.find_symbolic_locals_name(x) for x in kwargs['out'].items]
                    for idx, name in enumerate(output_tensor_names):
                        if name in tx.symbolic_locals:
                            tx.symbolic_locals[name] = tensor_variable.items[idx]
                elif isinstance(tensor_variable, TensorVariable):
                    assert isinstance(kwargs['out'], TensorVariable)
                    if kwargs['out'].source and kwargs['out'] in tx.output.graphargs and (kwargs['out'].size != tensor_variable.size):
                        unimplemented('out variants with resizing on graph inputs')
                    assert 'example_value' in kwargs['out'].proxy.node.meta
                    if not torch._prims_common.is_contiguous(kwargs['out'].proxy.node.meta['example_value']):
                        unimplemented('out= op was called where output tensor was non-contiguous')
                    name = tx.find_symbolic_locals_name(kwargs['out'])
                    if name in tx.symbolic_locals:
                        tx.symbolic_locals[name] = tensor_variable
                else:
                    unimplemented(f'out variant of {type(kwargs['out'])}')
            return tensor_variable

    def _call_ntuple(self, tx, args, kwargs):
        """inline behavior of torch.nn.modules.utils._ntuple"""
        if self.value is torch.nn.modules.utils._ntuple:
            count = args[0].as_python_constant()
        else:
            count = self.value.__closure__[0].cell_contents
        assert isinstance(count, int)
        assert not kwargs

        def handle_ntuple(value):
            if value.has_unpack_var_sequence(tx):
                return variables.TupleVariable(list(value.unpack_var_sequence(tx)))
            elif value.is_python_constant():
                return variables.ConstantVariable.create(torch.nn.modules.utils._ntuple(count)(value.as_python_constant()))
            else:
                unimplemented(f'torch.nn.modules.utils._ntuple({value})')
        if self.value is torch.nn.modules.utils._ntuple:
            return variables.LambdaVariable(handle_ntuple)
        else:
            return handle_ntuple(args[0])