import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _register_python_decomposition_vmap(decomp):
    if decomp in decomposition_table:
        VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
    else:
        raise RuntimeError(f'could not find decomposition for {decomp}')