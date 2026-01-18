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
@pytest.mark.parametrize('dim1', [1 * 2048])
@pytest.mark.parametrize('dim2', [2048])
@pytest.mark.parametrize('dtype', [torch.int8])
def test_spmm_coo_dequant(dim1, dim2, dtype):
    threshold = 6.0
    A = torch.randn(dim1, dim2, device='cuda').half()
    B = torch.empty(dim2, dim2 * 4, device='cuda', dtype=torch.float16)
    torch.nn.init.xavier_uniform_(B)
    Bt = B.t().contiguous()
    CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)
    rowidx = torch.randint(0, A.shape[-1], size=(15,))
    A[:, rowidx] = 8.0
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
    out1 = torch.matmul(A2, B.half())
    out3 = F.spmm_coo_very_sparse(cooA, CBt.half())
    out3 = out3 * statsBt.half() / 127
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    print(torch.median(max_count.float()))
    torch.testing.assert_close(out2, out3, rtol=0.05, atol=0.001)
    p = 200 / (2048 * 12288 * 4)
    n = out1.numel()
    count = math.ceil(p * n)
    assert_all_approx_close(out1, out2, rtol=0.01, atol=0.03, count=count)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    print('cusparse fp16', time.time() - t0)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = F.spmm_coo_very_sparse(cooA, CBt)
    torch.cuda.synchronize()
    print('int8', time.time() - t0)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
    torch.cuda.synchronize()
    print('int8+dequant', time.time() - t0)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = torch.matmul(A, B)
    torch.cuda.synchronize()
    print('matmul', time.time() - t0)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
        out = out1 + out2
    torch.cuda.synchronize()
    print('sparse+ matmul', time.time() - t0)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
        torch.matmul(A[:, rowidx], Bt.t()[rowidx], out=out1)
    torch.cuda.synchronize()
    print('partial matmul', time.time() - t0)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
    torch.cuda.synchronize()
    print('partial matmul', time.time() - t0)