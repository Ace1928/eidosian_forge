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
class CheckpointHigherOrderVariable(WrapHigherOrderVariable):

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn
        from .builder import wrap_fx_proxy
        context_fn = None
        if 'context_fn' in kwargs and kwargs['context_fn'] != noop_context_fn:
            context_fn = kwargs.pop('context_fn').fn
        checkpoint_kwargs, gmod_kwargs = TagActivationCheckpoint.divide_kwargs(kwargs)
        p_args, _, example_value, treespec, checkpointed_gmod = self.create_wrapped_node(tx, args, gmod_kwargs, 'torch.utils.checkpoint.checkpoint')
        if context_fn is not None:
            checkpointed_gmod.meta['_checkpoint_context_fn'] = context_fn
        _, checkpoint_kwargs = proxy_args_kwargs([], checkpoint_kwargs)
        variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs=checkpoint_kwargs), example_value=example_value)
        if treespec is None:
            return variable
        variable = BuiltinVariable(list).call_function(tx, [variable], {})
        return _make_inlined(tx, pytree.tree_unflatten)(variable, treespec)