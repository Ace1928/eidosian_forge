import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
    return (aten.addmm(b1, inp, w1), aten.addmm(b2, inp, w2), aten.addmm(b3, inp, w3))