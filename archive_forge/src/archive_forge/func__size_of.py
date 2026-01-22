from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def _size_of(node: fx.Node) -> int:
    if 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, py_sym_types):
            if isinstance(val, torch.SymInt):
                return 1
            else:
                return 999999
        elif isinstance(val, (list, tuple)):
            return sum((_tensor_nbytes(hint_int(n.numel(), fallback=4098), n.dtype) for n in val if isinstance(n, torch.Tensor)))
        elif isinstance(val, torch.Tensor):
            return _tensor_nbytes(hint_int(val.numel(), fallback=4098), val.dtype)
        raise RuntimeError(f'Unknown metadata type {type(val)}')
    if 'tensor_meta' in node.meta:
        metadata = node.meta['tensor_meta']
        numel = _prod(map(to_size_hint, metadata.shape))
        dtype = metadata.dtype
    else:
        return 0
    return _tensor_nbytes(numel, dtype)