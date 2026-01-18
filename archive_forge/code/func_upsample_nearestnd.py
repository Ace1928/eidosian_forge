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
def upsample_nearestnd(x, output_size, scales_x: Tuple[Optional[float], ...], n: int=2, exact: bool=False):
    x.realize_hint()
    x_loader = x.make_loader()
    i_sizes = x.get_size()[-n:]
    batch = x.get_size()[:-n]
    i_sizes = [V.graph.sizevars.evaluate_static_shape(i) for i in i_sizes]
    assert len(scales_x) == n
    o_sizes = output_size
    scales = [i / o for i, o in zip(i_sizes, o_sizes)]
    for i, scale in enumerate(scales_x):
        if scale:
            scales[i] = scale

    def scale_fn(x, scale, size):
        x = ops.index_expr(x, torch.float32)
        if exact:
            x = ops.add(x, ops.constant(0.5, torch.float32))
        x = ops.mul(x, ops.constant(scale, torch.float32))
        x = ops.to_dtype(x, torch.int32)
        return ops.indirect_indexing(x, size, check=False)

    def fn(idx):
        x = idx[-n:]
        b = idx[:-n]
        return x_loader([*b, *[scale_fn(i, s, size) for i, s, size in zip(x, scales, i_sizes)]])
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=[*batch, *o_sizes])