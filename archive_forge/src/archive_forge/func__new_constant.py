import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def _new_constant(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    assert isinstance(size, (list, tuple))
    assert_nyi(not pin_memory, 'pin_memory')
    assert_nyi(layout in (None, torch.strided), f'layout={layout}')
    dtype = decode_dtype(dtype) or x.get_dtype()
    device = device or x.get_device()
    size = [sympy.Integer(s) for s in size]
    return _full(fill_value, device, dtype, size)