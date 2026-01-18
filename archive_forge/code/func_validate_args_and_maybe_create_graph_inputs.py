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
def validate_args_and_maybe_create_graph_inputs(sub_args, tracer, tx, manually_set_subgraph_inputs):
    from . import AutogradFunctionContextVariable, ConstantVariable, SymNodeVariable, TensorVariable
    from .builder import wrap_fx_proxy, wrap_fx_proxy_cls
    assert tracer.parent is not None
    args = []
    for a in sub_args:
        assert isinstance(a, VariableTracker)
        if isinstance(a, ConstantVariable):
            if manually_set_subgraph_inputs:
                tracer.create_graph_input('const')
            new_arg = a
        elif isinstance(a, TensorVariable):
            if manually_set_subgraph_inputs:
                new_proxy = tracer.create_graph_input(a.as_proxy().node.name)
                example_value = a.as_proxy().node.meta['example_value']
                new_arg = wrap_fx_proxy(tx=tx, proxy=new_proxy, example_value=example_value)
            else:
                new_arg = a
        elif isinstance(a, SymNodeVariable):
            if manually_set_subgraph_inputs:
                new_proxy = tracer.create_graph_input(str(a.sym_num.node.expr))
                new_arg = wrap_fx_proxy_cls(target_cls=SymNodeVariable, tx=tx, proxy=new_proxy, example_value=a.sym_num)
            else:
                new_arg = a
        elif isinstance(a, AutogradFunctionContextVariable):
            if manually_set_subgraph_inputs:
                tracer.create_graph_input(a.as_proxy().node.name)
            new_arg = a
        elif manually_set_subgraph_inputs:
            raise unimplemented(f'HigherOrderOperator with body that accepts non-Tensors as input. Got: {a.python_type()}')
        elif only_consist_of(a, (ConstantVariable, SymNodeVariable, TensorVariable)):
            new_arg = a
        else:
            unimplemented("HigherOrderOperator with body that accepts non-Tensors as input that can't be lifted by tracer.")
        args.append(new_arg)
    return args