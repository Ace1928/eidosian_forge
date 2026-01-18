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
def use_two_step_variance(x, axis, keepdim):
    axis = _validate_reduction_axis(x, axis)
    kwargs = _make_reduction_inner(x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None)
    ranges = kwargs['ranges']
    reduction_numel = sympy_product(kwargs['reduction_ranges'])
    return isinstance(reduction_numel, sympy.Integer) and int(reduction_numel) < config.unroll_reductions_threshold and (sympy_product(ranges) != 1)