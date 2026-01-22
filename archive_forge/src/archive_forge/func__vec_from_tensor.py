import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _vec_from_tensor(x, generator, downcast_complex=False):
    if x.layout == torch.sparse_coo:
        x_values = x._values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = torch.rand(x_values.numel(), generator=generator).to(dtype=dtype, device=x.device).view(x_values.shape)
        values /= values.norm()
        vec = torch.sparse_coo_tensor(x._indices(), values, x.size(), device=x.device)
    elif _is_sparse_compressed_tensor(x):
        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices, plain_indices = (x.crow_indices(), x.col_indices())
        else:
            compressed_indices, plain_indices = (x.ccol_indices(), x.row_indices())
        x_values = x.values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = torch.rand(x_values.numel(), generator=generator).to(dtype=dtype, device=x.device).view(x_values.shape)
        values /= values.norm()
        vec = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, x.size(), layout=x.layout, device=x.device)
    else:
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        vec = torch.rand(x.numel(), generator=generator).to(dtype=dtype, device=x.device)
        vec /= vec.norm()
    return vec