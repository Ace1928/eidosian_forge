import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def _unwrap(tensor):
    if isinstance(tensor, triton.TensorWrapper):
        return tensor.base
    return tensor