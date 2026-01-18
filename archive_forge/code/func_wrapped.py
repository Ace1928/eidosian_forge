import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def wrapped(*args, **kwargs):
    ret = fn(*args, **kwargs)
    return TensorHandle(ret.data, compute_ret_ty(*args, **kwargs))