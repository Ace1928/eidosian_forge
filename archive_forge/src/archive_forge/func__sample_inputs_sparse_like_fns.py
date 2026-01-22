import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _sample_inputs_sparse_like_fns(op_info, device, dtype, requires_grad, layout, **kwargs):
    from torch.testing._internal.common_utils import TestCase
    for tensor in TestCase().generate_simple_inputs(layout, device=device, dtype=dtype, enable_batch=True, enable_hybrid=True, enable_zero_sized=True, enable_non_contiguous_indices=False, enable_non_contiguous_values=False):
        yield SampleInput(tensor, args=(), kwargs={})
        yield SampleInput(tensor, args=(), kwargs=dict(device=device, dtype=dtype, layout=layout))
        if dtype is not torch.float64:
            yield SampleInput(tensor, args=(), kwargs=dict(dtype=torch.float64))
        if torch.cuda.is_available():
            other_device = 'cuda' if tensor.device.type == 'cpu' else 'cpu'
            yield SampleInput(tensor, args=(), kwargs=dict(device=other_device))
        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided
        yield SampleInput(tensor, args=(), kwargs=dict(layout=other_layout))
        if layout is not torch.sparse_coo:
            yield SampleInput(tensor, args=(), kwargs=dict(layout=torch.sparse_coo))