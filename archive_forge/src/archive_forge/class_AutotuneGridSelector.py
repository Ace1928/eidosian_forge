import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
class AutotuneGridSelector:

    def __init__(self, func, autotune_params):
        self.func = func
        self.autotune_params = autotune_params

    def __getitem__(self, grid):
        return AutotuneRunner(self.func, self.autotune_params, grid)

    def __call__(self, *args, **kwargs):
        return AutotuneRunner(self.func, self.autotune_params)(*args, **kwargs)