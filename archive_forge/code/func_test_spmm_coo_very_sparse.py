from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
@pytest.mark.parametrize('dim1', [1 * 2048], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [12288], ids=id_formatter('dim2'))
@pytest.mark.parametrize('dtype', [torch.float16], ids=describe_dtype)
@pytest.mark.parametrize('out_func', ['zeros', 'ones'], ids=id_formatter('out_func'))
def test_spmm_coo_very_sparse(dim1, dim2, dtype, out_func):
    out_func = getattr(torch, out_func)
    threshold = 3.3
    A = torch.randn(dim1, dim2, device='cuda').half()
    if dtype == torch.float16:
        B = torch.randn(dim2, dim2 * 4, device='cuda').half()
        torch.nn.init.xavier_uniform_(B)
    else:
        B = torch.randn(dim2, dim2 * 4, device='cuda').half()
        torch.nn.init.xavier_uniform_(B)
        B, SB = F.vectorwise_quant(B, quant_type='linear')
    print('')
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    out1 = torch.matmul(A2.half(), B.half())
    out = out_func(out1.shape, dtype=torch.float16, device=out1.device)
    out1 += out.clone()
    out2 = F.spmm_coo_very_sparse(cooA, B, out=out)
    p = 200 / (2048 * 12288 * 4)
    n = out1.numel()
    count = math.ceil(p * n)
    std = out1.std()
    out1 /= std
    out2 /= std
    assert_all_approx_close(out1, out2.half(), rtol=0.01, atol=0.03, count=count)
    idx_col = torch.randint(0, A2.shape[-1], size=(15,))