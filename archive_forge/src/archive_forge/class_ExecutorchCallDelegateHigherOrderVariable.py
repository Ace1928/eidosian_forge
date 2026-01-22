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
class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from .builder import wrap_fx_proxy
        if len(kwargs) > 0:
            unimplemented('executorch_call_delegate: kwargs arguments were not enabled.')
        lowered_module = tx.output.get_submodule(args[0].module_key)
        lowered_node = make_attr(tx, args[0].module_key)
        p_args = tuple((arg.as_proxy() for arg in args[1:]))
        real_sub_args = pytree.tree_map_only(torch.fx.Proxy, lambda a: get_real_value(a.node, tx.output), p_args)
        example_res = lowered_module.original_module(*real_sub_args)
        _assert_tensors_nonaliasing(real_sub_args, example_res)
        example_value = deepcopy_to_fake_tensor(example_res, tx.fake_mode)
        p_args = (lowered_node,) + p_args
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)