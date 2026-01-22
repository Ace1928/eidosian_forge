from __future__ import annotations
import itertools
import logging
import weakref
from typing import Any, List, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code
from torch._functorch.aot_autograd import MutationType
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape
from . import config
class ErasedTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device='meta'))

    def __init__(self, elem, name: Optional[str], mod):
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        erased_tensors = [e for e in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(e, ErasedTensor)]
        assert len(erased_tensors) > 0
        e = erased_tensors[0]
        raise RuntimeError(f'Trying to run Pytorch Eager Module after Dynamo Freezing. The original parameters have been discarded for memory efficiency. Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}')