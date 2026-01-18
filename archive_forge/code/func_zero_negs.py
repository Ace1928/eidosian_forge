import functools
import unittest
from torch.testing._internal.inductor_utils import HAS_CUDA
@triton.jit
def zero_negs(x):
    return tl.where(x >= 0, x, 0)