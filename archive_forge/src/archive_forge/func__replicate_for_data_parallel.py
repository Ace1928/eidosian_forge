import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
def _replicate_for_data_parallel(self: _LazyProtocol):
    raise RuntimeError("Modules with uninitialized parameters can't be used with `DataParallel`. Run a dummy forward pass to correctly initialize the modules")