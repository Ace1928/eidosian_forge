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
@pytest.mark.benchmark
def test_spmm_bench():
    batch = 2
    model = 1024 * 1
    hidden = model * 4
    seq = 1024
    dim1 = batch * seq
    dim2 = model
    dim3 = hidden
    threshold = 4
    A = torch.randn(dim1, dim2, device='cuda').half()
    B = torch.randn(dim2, dim3, device='cuda').half()
    for i in range(10):
        C1 = bnb.matmul(A, B.t())
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        C1 = bnb.matmul(A, B.t())
    torch.cuda.synchronize()
    t8 = time.time() - t0
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    print(nnz / idx.numel())
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    for i in range(10):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    tsp = time.time() - t0
    print(tsp, t8)
    print(tsp / t8)