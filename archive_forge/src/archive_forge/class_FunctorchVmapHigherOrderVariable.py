import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
from ..exc import (
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
class FunctorchVmapHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable, TensorVariable
        from .builder import wrap_fx_proxy
        if not torch._dynamo.config.capture_func_transforms:
            unimplemented('torch.func.vmap capture is disabled, it can be turned on by setting `torch._dynamo.config.capture_func_transforms=True`')
        checkpoint = tx.copy_graphstate()
        graph_checkpoint = tx.output.graph
        fn = args[0]
        in_dims = args[1]
        out_dims = args[2]
        randomness = args[3]
        chunk_size = args[4]
        batch_input_args = args[5:]
        if not isinstance(in_dims, (ConstantVariable, TupleVariable)):
            unimplemented('torch.func.vmap: in_dims is not an int or tuple variable.')
        if not isinstance(out_dims, (ConstantVariable, TupleVariable)):
            unimplemented('torch.func.vmap: out_dims is not an int or tuple variable.')
        if len(kwargs) > 0:
            unimplemented('NYI - torch.func.vmap: kwargs arguments are currently unsupported.')
        if chunk_size.value is not None:
            unimplemented('NYI - torch.func.vmap is not implemented when chunk_size is passed')
        flat_args, arg_spec = _make_inlined(tx, pytree.tree_flatten)(ListVariable(batch_input_args)).unpack_var_sequence(tx)
        in_dims_v = in_dims if isinstance(in_dims.as_python_constant(), int) else BuiltinVariable(list).call_function(tx, [in_dims], {})
        broadcasted_in_dims = _make_inlined(tx, pytree._broadcast_to_and_flatten)(in_dims_v, arg_spec)
        unbatched_input_args = []
        for arg, in_dim in zip(flat_args.unpack_var_sequence(tx), broadcasted_in_dims.unpack_var_sequence(tx)):
            if in_dim is not None:
                assert isinstance(arg, TensorVariable)
                unbatched_arg = arg.call_method(tx, 'select', [in_dim, ConstantVariable.create(0)], {})
                unbatched_input_args.append(unbatched_arg)
            else:
                unbatched_input_args.append(arg)
        with tx.strict_translation_mode():
            _, body_graph, body_lifted_freevars = speculate_subgraph(tx, fn, _make_inlined(tx, pytree.tree_unflatten)(ListVariable(unbatched_input_args), arg_spec).unpack_var_sequence(tx), {}, graph_checkpoint, checkpoint, 'torch.vmap', source_target=self.value)
        body_name = add_subgraph(tx, self.source, 'vmap_body', torch.fx.GraphModule(tx.output.nn_modules, body_graph))
        body_node = make_attr(tx, body_name)
        updated_in_dims = TupleVariable(broadcasted_in_dims.unpack_var_sequence(tx) + [ConstantVariable.create(None)] * len(body_lifted_freevars))
        vmap_proxy_args = (body_node, *(arg.as_proxy() for arg in (updated_in_dims, out_dims, randomness)))
        vmap_proxy = tx.output.create_proxy('call_function', torch.func.vmap, args=tuple(vmap_proxy_args), kwargs={}, name='vmap_proxy')
        proxy_batched_fn_args = tuple((arg.as_proxy() for arg in batch_input_args)) + tuple(body_lifted_freevars)
        fake_batched_fn_args = itertools.chain((get_fake_value(arg.as_proxy().node, tx) for arg in batch_input_args), (get_fake_value(arg.node, tx) for arg in body_lifted_freevars))
        actual_in_dims = tuple(pytree.tree_map(lambda x: x.value, updated_in_dims.items))
        with tx.fake_mode, enable_python_dispatcher():
            example_value = torch._functorch.vmap.vmap_impl(torch.fx.GraphModule(tx.output.nn_modules, body_graph), actual_in_dims, out_dims.as_python_constant(), randomness.value, chunk_size.value, *fake_batched_fn_args)
        proxy = vmap_proxy(*proxy_batched_fn_args)
        return wrap_fx_proxy(tx=tx, proxy=proxy, example_value=example_value)