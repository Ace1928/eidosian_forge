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
@register_lowering(aten.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(x, output_size=None, input_size=None, scales_h=None, scales_w=None):
    x.realize_hint()
    *batch, inp_h, inp_w = x.get_size()
    inp_h = V.graph.sizevars.evaluate_static_shape(inp_h)
    inp_w = V.graph.sizevars.evaluate_static_shape(inp_w)
    *batch, out_h, out_w = input_size
    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(x, [inp_h // out_h, inp_w // out_w], divisor_override=1)
    h_kernel_max = ceildiv(inp_h, out_h)
    w_kernel_max = ceildiv(inp_w, out_w)

    def start_index(index, out_dim, inp_dim):
        return CeilDiv(index * inp_dim, out_dim)

    def end_index(index, out_dim, inp_dim):
        return start_index(index + 1, out_dim, inp_dim)
    h_start_index = functools.partial(start_index, out_dim=out_h, inp_dim=inp_h)
    h_end_index = functools.partial(end_index, out_dim=out_h, inp_dim=inp_h)
    w_start_index = functools.partial(start_index, out_dim=out_w, inp_dim=inp_w)
    w_end_index = functools.partial(end_index, out_dim=out_w, inp_dim=inp_w)
    fn_sum = _adaptive_pooling_idx_sum([h_kernel_max, w_kernel_max], [h_start_index, w_start_index], [h_end_index, w_end_index])

    def fn(idx):
        return fn_sum(idx, pad_adaptive_loader(x))
    rv = Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=list(input_size))
    return rv