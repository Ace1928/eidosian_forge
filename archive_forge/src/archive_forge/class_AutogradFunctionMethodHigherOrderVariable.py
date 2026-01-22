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
class AutogradFunctionMethodHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def __init__(self, value, fwd_bwd_tracer=None, source: Optional[Source]=None, **kwargs):
        super().__init__(value, source, **kwargs)
        self.fwd_bwd_tracer = fwd_bwd_tracer

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import UserFunctionVariable
        from .builder import wrap_fx_proxy
        tracer = self.fwd_bwd_tracer
        if len(kwargs) > 0:
            unimplemented('kwargs have not been implemented for torch.autograd.Function')
        from . import TorchVariable
        always_restore = self.value.__name__ == 'trampoline_autograd_bwd'
        if self.value.__name__ == 'trampoline_autograd_bwd' or self.value.__name__ == 'trampoline_autograd_fwd':
            fn = UserFunctionVariable(self.value, source=self.source)
        else:
            fn = TorchVariable(self.value)
        checkpoint = tx.copy_graphstate()
        pre_guards = tx.output.guards
        graph_checkpoint = tx.output.graph
        enable_grad = False if self.value.__name__ == 'trampoline_autograd_bwd' else None
        (body_r, _), body_graph, body_lifted_freevars = speculate_subgraph(tx, fn, [*args], {}, graph_checkpoint, checkpoint, 'the user-defined autograd.Function', source_target=self.value, always_restore=always_restore, restore_side_effects=False, tracer=tracer, enable_grad=enable_grad)
        post_guards = tx.output.guards
        if body_lifted_freevars:
            unimplemented('NYI - freevars in autograd function.')
        if always_restore:
            if post_guards - pre_guards:
                unimplemented('NYI - New guards discovered in a restoring state')
            return None
        p_args = (*(arg.as_proxy() for arg in args), *(arg for arg in body_lifted_freevars.keys()))
        example_value = pytree.tree_map_only(torch.fx.Proxy, lambda a: a.node.meta['example_value'], body_r.as_proxy())
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)