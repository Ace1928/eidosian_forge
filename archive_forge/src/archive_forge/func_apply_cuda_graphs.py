import logging
import operator
from collections import defaultdict
from typing import Set
import torch
from torch.fx import GraphModule
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn import Module
from torch.utils._pytree import tree_map
from .common import aot_autograd
from .registry import register_backend
def apply_cuda_graphs(gm):
    for n in gm.graph.nodes:
        if n.op == 'call_module':
            assert not n.kwargs
            submod = gm.get_submodule(n.target)
            gm.delete_submodule(n.target)
            mutated_inputs = find_input_mutations(submod.graph)
            gm.add_submodule(n.target, CudaGraphModule(submod, mutated_inputs))