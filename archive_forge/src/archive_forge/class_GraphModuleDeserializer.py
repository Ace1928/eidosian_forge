import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
class GraphModuleDeserializer:

    @dataclasses.dataclass
    class Result:
        graph_module: torch.fx.GraphModule
        signature: ep.ExportGraphSignature
        module_call_graph: List[ep.ModuleCallEntry]
        names_to_symbols: Dict[str, sympy.Symbol]

    def __init__(self):
        self.serialized_name_to_node: Dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: Dict[str, MetaType] = {}
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()

    @contextmanager
    def save_graph_module(self) -> Iterator[None]:
        saved = (self.graph, self.module, self.serialized_name_to_node, self.serialized_name_to_meta)
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()
        self.serialized_name_to_node = {}
        self.serialized_name_to_meta = {}
        try:
            yield
        finally:
            self.graph, self.module, self.serialized_name_to_node, self.serialized_name_to_meta = saved

    def deserialize_operator(self, serialized_target: str):
        if serialized_target.startswith('_operator'):
            module = operator
            serialized_target_names = serialized_target.split('.')[1:]
        elif serialized_target.startswith('torch.ops'):
            module = torch.ops
            serialized_target_names = serialized_target.split('.')[2:]
        else:
            return serialized_target
        target = module
        for name in serialized_target_names:
            if not hasattr(target, name):
                return serialized_target
            else:
                target = getattr(target, name)
        return target

    def deserialize_sym_int(self, s: SymInt) -> Union[int, torch.SymInt]:
        val = s.value
        if s.type == 'as_expr':
            if val.expr_str in self.symbol_name_to_symbol:
                sym = self.symbol_name_to_symbol[val.expr_str]
            else:
                sym = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
                if isinstance(sym, sympy.Symbol):
                    self.symbol_name_to_symbol[val.expr_str] = sym
                    if (vr := self.symbol_name_to_range.get(val.expr_str)):
                        symbolic_shapes._constrain_symbol_range(self.shape_env, sym, compiler_min=vr.lower, compiler_max=vr.upper, runtime_min=vr.lower, runtime_max=vr.upper)
            if val.hint is None:
                hint = None
            else:
                assert val.hint.type == 'as_int'
                hint = val.hint.value
            return self.shape_env.create_symintnode(sym, hint=hint)
        elif s.type == 'as_int':
            assert isinstance(val, int)
            return val
        else:
            raise SerializeError(f'SymInt has invalid field type {s.type} with value {s.value}')

    def deserialize_sym_bool(self, s: SymBool) -> Union[bool, torch.SymBool]:
        val = s.value
        if s.type == 'as_expr':
            expr = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
            return self.shape_env.create_symboolnode(expr)
        elif s.type == 'as_bool':
            assert isinstance(val, bool)
            return val
        else:
            raise SerializeError(f'SymBool has invalid field type {s.type} with value {s.value}')

    def deserialize_tensor_meta(self, tensor_meta: TensorMeta, fake_tensor_mode: FakeTensorMode) -> FakeTensor:
        with fake_tensor_mode:
            return cast(FakeTensor, torch.empty_strided(tuple((self.deserialize_sym_int(val) for val in tensor_meta.sizes)), tuple((self.deserialize_sym_int(val) for val in tensor_meta.strides)), device=deserialize_device(tensor_meta.device), dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype]))

    def deserialize_graph_output(self, output) -> torch.fx.Node:
        if isinstance(output.value, TensorArgument):
            return self.serialized_name_to_node[output.value.name]
        elif isinstance(output.value, (SymIntArgument, SymBoolArgument)):
            return self.serialized_name_to_node[output.value.as_name]
        else:
            raise SerializeError(f'Unable to deserialize output node {output}')

    def deserialize_graph(self, serialized_graph: Graph) -> torch.fx.Graph:
        for name, tensor_value in serialized_graph.tensor_values.items():
            meta_val = self.deserialize_tensor_meta(tensor_value, self.fake_tensor_mode)
            self.serialized_name_to_meta[name] = meta_val
        for name, sym_int_value in serialized_graph.sym_int_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_sym_int(sym_int_value)
        for name, sym_bool_value in serialized_graph.sym_bool_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_sym_bool(sym_bool_value)
        for input in serialized_graph.inputs:
            placeholder_node = self.graph.placeholder(input.as_tensor.name)
            self.sync_fx_node(input.as_tensor.name, placeholder_node)
        for serialized_node in serialized_graph.nodes:
            try:
                target = self.deserialize_operator(serialized_node.target)
                self.deserialize_node(serialized_node, target)
            except Exception as e:
                raise SerializeError(f'Failed deserializing node {serialized_node}') from e
        outputs = []
        for output in serialized_graph.outputs:
            outputs.append(self.deserialize_graph_output(output))
        if serialized_graph.is_single_tensor_return:
            assert len(outputs) == 1
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        output_node = self.graph.output(outputs)
        if serialized_graph.is_single_tensor_return:
            output_node.meta['val'] = output_node.args[0].meta['val']
        else:
            output_node.meta['val'] = tuple((arg.meta['val'] for arg in output_node.args[0]))
        return self.graph

    def deserialize_node(self, serialized_node: Node, target: Callable) -> None:
        if target.__module__ == '_operator':
            name = serialized_node.outputs[0].value.as_name
            args = self.deserialize_sym_op_inputs(serialized_node.inputs)
            fx_node = self.graph.create_node('call_function', target, args, {}, name)
            self.deserialize_sym_op_outputs(serialized_node, fx_node)
        elif isinstance(target, torch._ops.HigherOrderOperator):
            assert len(serialized_node.outputs) == 1 and serialized_node.outputs[0].type in ('as_tensors', 'as_tensor'), 'Only single tensor output or list of tensor output is supported for higher order operators.'
            output = serialized_node.outputs[0]
            name = output.value.name if output.type == 'as_tensor' else None
            args = tuple((self.deserialize_input(input.arg) for input in serialized_node.inputs))
            fx_node = self.graph.create_node('call_function', target, args, {}, name)
            if output.type == 'as_tensor':
                self.sync_fx_node(name, fx_node)
            if output.type == 'as_tensors':
                self.deserialize_multiple_outputs(serialized_node, fx_node)
        elif isinstance(target, torch._ops.OpOverload):
            name = serialized_node.outputs[0].value.name if _is_single_tensor_return(target) else None
            args, kwargs = self.deserialize_inputs(target, serialized_node)
            fx_node = self.graph.create_node('call_function', target, args, kwargs, name)
            self.deserialize_outputs(serialized_node, fx_node)
        else:
            raise SerializeError(f'Unsupported target type for node {serialized_node}: {target}')
        fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))

    def deserialize_input_spec(self, i: InputSpec) -> ep.InputSpec:
        if i.user_input is not None:
            return ep.InputSpec(kind=ep.InputKind.USER_INPUT, arg=self.deserialize_argument_spec(i.user_input.arg), target=None)
        elif i.parameter is not None:
            return ep.InputSpec(kind=ep.InputKind.PARAMETER, arg=PyTensorArgument(name=i.parameter.arg.name), target=i.parameter.parameter_name)
        elif i.buffer is not None:
            return ep.InputSpec(kind=ep.InputKind.BUFFER, arg=PyTensorArgument(name=i.buffer.arg.name), target=i.buffer.buffer_name)
        elif i.tensor_constant is not None:
            return ep.InputSpec(kind=ep.InputKind.CONSTANT_TENSOR, arg=PyTensorArgument(name=i.tensor_constant.arg.name), target=i.tensor_constant.tensor_constant_name)
        else:
            raise AssertionError(f'Unkown input spec {i}')

    def deserialize_output_spec(self, o: OutputSpec) -> ep.OutputSpec:
        if o.user_output is not None:
            return ep.OutputSpec(kind=ep.OutputKind.USER_OUTPUT, arg=self.deserialize_argument_spec(o.user_output.arg), target=None)
        elif o.loss_output is not None:
            return ep.OutputSpec(kind=ep.OutputKind.LOSS_OUTPUT, arg=PyTensorArgument(name=o.loss_output.arg.name), target=None)
        elif o.buffer_mutation is not None:
            return ep.OutputSpec(kind=ep.OutputKind.BUFFER_MUTATION, arg=PyTensorArgument(name=o.buffer_mutation.arg.name), target=o.buffer_mutation.buffer_name)
        elif o.gradient_to_parameter is not None:
            return ep.OutputSpec(kind=ep.OutputKind.GRADIENT_TO_PARAMETER, arg=PyTensorArgument(name=o.gradient_to_parameter.arg.name), target=o.gradient_to_parameter.parameter_name)
        elif o.gradient_to_user_input is not None:
            return ep.OutputSpec(kind=ep.OutputKind.GRADIENT_TO_USER_INPUT, arg=PyTensorArgument(name=o.gradient_to_user_input.arg.name), target=o.gradient_to_user_input.user_input_name)
        else:
            raise AssertionError(f'Unknown output spec {o}')

    def deserialize_signature(self, sig: GraphSignature) -> ep.ExportGraphSignature:
        return ep.ExportGraphSignature(input_specs=[self.deserialize_input_spec(i) for i in sig.input_specs], output_specs=[self.deserialize_output_spec(o) for o in sig.output_specs])

    def deserialize(self, serialized_graph_module: GraphModule, symbol_name_to_range: Optional[Dict[str, symbolic_shapes.ValueRanges]]=None, constants: Optional[Dict[str, Any]]=None) -> Result:
        self.shape_env = symbolic_shapes.ShapeEnv(assume_static_by_default=True)
        self.fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True, shape_env=self.shape_env)
        self.symbol_name_to_symbol: Dict[str, sympy.Symbol] = {}
        self.symbol_name_to_range = {} if symbol_name_to_range is None else symbol_name_to_range
        self.constants = {} if constants is None else constants
        self.deserialize_graph(serialized_graph_module.graph)
        sig = self.deserialize_signature(serialized_graph_module.signature)
        module_call_graph = self.deserialize_module_call_graph(serialized_graph_module.module_call_graph)
        return GraphModuleDeserializer.Result(graph_module=torch._export.exported_program._create_graph_module_for_export(self.module, self.graph), signature=sig, module_call_graph=module_call_graph, names_to_symbols=self.symbol_name_to_symbol)

    def sync_fx_node(self, name: str, fx_node: torch.fx.Node):
        if name in self.serialized_name_to_node:
            raise SerializeError(f'Node {name} has already been deserialized before.')
        self.serialized_name_to_node[name] = fx_node
        assert 'val' not in fx_node.meta
        fx_node.meta['val'] = self.serialized_name_to_meta[name]

    def deserialize_sym_op_inputs(self, inputs):
        return tuple((self.deserialize_input(input.arg) for input in inputs))

    def deserialize_inputs(self, target: torch._ops.OpOverload, serialized_node: Node):
        schema_args = target._schema.arguments
        actual_args = {input.name: self.deserialize_input(input.arg) for input in serialized_node.inputs}
        args = []
        kwargs = {}
        for schema_arg in schema_args:
            is_positional = not schema_arg.has_default_value() and (not schema_arg.kwarg_only)
            if is_positional:
                args.append(actual_args[schema_arg.name])
            elif schema_arg.name in actual_args:
                kwargs[schema_arg.name] = actual_args[schema_arg.name]
        return (tuple(args), kwargs)

    def deserialize_input(self, inp: Argument) -> Any:
        value = inp.value
        typ_ = inp.type
        if typ_ == 'as_none':
            return None
        elif typ_ == 'as_scalar_type':
            return _SERIALIZE_TO_TORCH_DTYPE[value]
        elif typ_ == 'as_memory_format':
            return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[value]
        elif typ_ == 'as_layout':
            return _SERIALIZE_TO_TORCH_LAYOUT[value]
        elif typ_ == 'as_graph':
            assert isinstance(value, GraphArgument)
            with self.save_graph_module():
                self.deserialize_graph(value.graph)
                submodule = torch._export.exported_program._create_graph_module_for_export(self.module, self.graph)
            self.module.register_module(value.name, submodule)
            return self.graph.create_node('get_attr', value.name, name=value.name)
        elif isinstance(value, Device):
            return deserialize_device(value)
        elif isinstance(value, TensorArgument):
            return self.serialized_name_to_node[value.name]
        elif isinstance(value, (int, float, bool)):
            return value
        elif isinstance(value, str):
            return str(value)
        elif isinstance(value, (SymIntArgument, SymBoolArgument)):
            return self.deserialize_sym_argument(value)
        elif isinstance(value, list):
            if len(value) == 0:
                return []
            elif isinstance(value[0], TensorArgument):
                result = []
                for arg in value:
                    result.append(self.serialized_name_to_node[arg.name])
                return result
            elif isinstance(value[0], (int, float, bool)):
                return list(value)
            elif isinstance(value[0], (SymIntArgument, SymBoolArgument)):
                return [self.deserialize_sym_argument(arg) for arg in value]
            elif isinstance(value[0], OptionalTensorArgument):

                def deserialize_optional_tensor_args(a):
                    if a.type == 'as_none':
                        return None
                    elif a.type == 'as_tensor':
                        return self.serialized_name_to_node[a.value]
                    else:
                        raise SerializeError(f'Unhandled argument {inp}')
                return list(map(deserialize_optional_tensor_args, value))
            else:
                raise SerializeError(f'Unhandled argument {inp}')
        elif isinstance(value, CustomObjArgument):
            return self.constants[value.name]
        else:
            raise SerializeError(f'Unhandled argument {inp}')

    def deserialize_sym_argument(self, sym_int_arg):
        if sym_int_arg.type == 'as_int':
            return sym_int_arg.as_int
        else:
            assert sym_int_arg.type == 'as_name'
            return self.serialized_name_to_node[sym_int_arg.as_name]

    def deserialize_sym_op_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)

    def deserialize_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        assert isinstance(fx_node.target, torch._ops.OpOverload)
        returns = fx_node.target._schema.returns
        if len(returns) == 0:
            return
        if _is_single_tensor_return(fx_node.target):
            self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
            return
        elif len(returns) == 1 and isinstance(serialized_node.outputs[0].value, (SymIntArgument, SymBoolArgument)):
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return
        self.deserialize_multiple_outputs(serialized_node, fx_node)

    def deserialize_multiple_outputs(self, serialized_node: Node, fx_node: torch.fx.Node) -> None:
        deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)

        def generate_getitem(meta_val, fx_node: torch.fx.Node, arg: TensorArgument, idx: int):
            name = arg.name
            individual_output = self.graph.create_node('call_function', operator.getitem, (fx_node, idx), name=name)
            self.sync_fx_node(name, individual_output)
            meta_val.append(self.serialized_name_to_meta[name])
            individual_output.meta.update(deserialized_metadata)

        def generate_getitems(meta_val, fx_node: torch.fx.Node, args):
            for idx, arg in enumerate(args):
                if isinstance(arg, Argument):
                    arg = arg.value
                if isinstance(arg, TensorArgument):
                    generate_getitem(meta_val, fx_node, arg, idx)
                elif isinstance(arg, (list, tuple)):
                    list_output = self.graph.create_node('call_function', operator.getitem, (fx_node, idx))
                    meta_val.append([])
                    generate_getitems(meta_val[-1], list_output, arg)
                    list_output.meta.update(deserialized_metadata)
                    list_output.meta['val'] = meta_val[-1]
                else:
                    raise NotImplementedError(f'Unimplemented node output type: {arg}')
        meta_val: List[Any] = []
        if len(serialized_node.outputs) == 1:
            assert isinstance(serialized_node.outputs[0].value, list)
            assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
            generate_getitems(meta_val, fx_node, serialized_node.outputs[0].as_tensors)
        else:
            generate_getitems(meta_val, fx_node, serialized_node.outputs)
        fx_node.meta['val'] = tuple(meta_val)
        self.serialized_name_to_node[fx_node.name] = fx_node

    def deserialize_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        ret: Dict[str, Any] = {}
        if (stack_trace := metadata.get('stack_trace')):
            ret['stack_trace'] = stack_trace

        def deserialize_meta_func(serialized_target: str):
            module = None
            if serialized_target.startswith('torch.nn'):
                module = torch.nn
                serialized_target_names = serialized_target.split('.')[2:]
            elif serialized_target.startswith('torch'):
                module = torch
                serialized_target_names = serialized_target.split('.')[1:]
            else:
                return self.deserialize_operator(serialized_target)
            target = module
            for name in serialized_target_names:
                if not hasattr(target, name):
                    return serialized_target
                else:
                    target = getattr(target, name)
            return target
        if (nn_module_stack_str := metadata.get('nn_module_stack')):

            def import_nn_module_stack(key, path, ty):
                return (key, (path, ty))
            nn_module_stack = dict((import_nn_module_stack(*item.split(',')) for item in nn_module_stack_str.split(ST_DELIMITER)))
            ret['nn_module_stack'] = nn_module_stack
        if (source_fn_st_str := metadata.get('source_fn_stack')):
            source_fn_st = []
            for source_fn_str in source_fn_st_str.split(ST_DELIMITER):
                name, target_str = source_fn_str.split(',')
                source_fn_st.append((name, deserialize_meta_func(target_str)))
            ret['source_fn_stack'] = source_fn_st
        return ret

    def deserialize_argument_spec(self, x: Argument) -> ep.ArgumentSpec:
        if x.as_tensor is not None:
            return PyTensorArgument(name=x.as_tensor.name)
        elif x.as_sym_int is not None:
            return PySymIntArgument(name=x.as_sym_int.as_name)
        else:
            return PyConstantArgument(value=self.deserialize_input(x))

    def deserialize_module_call_signature(self, module_call_signature: ModuleCallSignature) -> ep.ModuleCallSignature:
        return ep.ModuleCallSignature(inputs=[self.deserialize_argument_spec(x) for x in module_call_signature.inputs], outputs=[self.deserialize_argument_spec(x) for x in module_call_signature.outputs], in_spec=treespec_loads(module_call_signature.in_spec), out_spec=treespec_loads(module_call_signature.out_spec))

    def deserialize_module_call_graph(self, module_call_graph: List[ModuleCallEntry]) -> List[ep.ModuleCallEntry]:
        return [ep.ModuleCallEntry(fqn=entry.fqn, signature=self.deserialize_module_call_signature(entry.signature) if entry.signature else None) for entry in module_call_graph]