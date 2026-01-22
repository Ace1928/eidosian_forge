from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _grad_postprocess(inputs, create_graph):
    if isinstance(inputs[0], torch.Tensor):
        if not create_graph:
            return tuple((inp.detach() for inp in inputs))
        else:
            return inputs
    else:
        return tuple((_grad_postprocess(inp, create_graph) for inp in inputs))