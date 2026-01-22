import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
@classmethod
def advance_offset(cls, consumed_offset):
    cls.running_state.advance_offset(consumed_offset)