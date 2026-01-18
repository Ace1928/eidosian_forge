import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def unary_op(self, arg, op):
    return TensorHandle(op(arg.data), arg.dtype)