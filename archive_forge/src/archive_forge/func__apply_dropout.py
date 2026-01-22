import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def _apply_dropout(att, dropout):
    if dropout is None:
        return att
    if _has_cpp_library:
        if isinstance(att, SparseCS):
            values = att.values.clone()
            values = dropout(values)
            att = SparseCS.wrap(att.shape, values, att.row_indices, att.row_offsets, att.column_indices, att._transp_info)
        elif att.is_sparse:
            att = att.coalesce()
            values = att.values().clone()
            values = dropout(values)
            att = torch.sparse_coo_tensor(att.indices(), values, att.shape)
        else:
            att = dropout(att)
        return att
    att = dropout(att)
    return att