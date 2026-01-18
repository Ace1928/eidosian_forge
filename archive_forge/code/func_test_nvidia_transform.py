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
@pytest.mark.parametrize('dim1', get_test_dims(2, 256, n=2), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', get_test_dims(2, 256, n=2), ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', get_test_dims(2, 256, n=2), ids=id_formatter('dim3'))
@pytest.mark.parametrize('dtype', [torch.int8, torch.int32], ids=describe_dtype)
@pytest.mark.parametrize('orderA', ['row'], ids=id_formatter('orderA'))
@pytest.mark.parametrize('orderOut', ['col', 'row', 'col32'], ids=id_formatter('orderOut'))
@pytest.mark.parametrize('transpose', [False], ids=id_formatter('transpose'))
@pytest.mark.parametrize('dims', [2, 3], ids=id_formatter('dims'))
def test_nvidia_transform(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    if dims == 3 and orderOut != 'col32':
        return
    if dtype == torch.int32 and orderOut != 'col32':
        return
    try:
        func = F.get_transform_func(dtype, orderA, orderOut, transpose)
    except ValueError as ve:
        pytest.skip(str(ve))
    if dims == 2:
        A = torch.randint(-128, 127, size=(dim1, dim2), device='cuda').to(dtype)
    elif dims == 3:
        A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device='cuda').to(dtype)
    out, S = F.nvidia_transform(A, to_order=orderOut)
    if orderOut == 'row':
        torch.testing.assert_close(A.flatten(), out.flatten())
    elif orderOut == 'col':
        torch.testing.assert_close(A.t().flatten(), out.flatten())
    elif orderOut == 'col32':
        if dims == 2:
            n = A.shape[0] * (A.shape[1] + (32 - A.shape[1] % 32))
        elif dims == 3:
            n = A.shape[0] * A.shape[1] * (A.shape[2] + (32 - A.shape[2] % 32))
        assert out.numel() == n
    elif orderOut == 'col_turing':
        n = (A.shape[0] + (8 - A.shape[0] % 8)) * (A.shape[1] + (32 - A.shape[1] % 32))
        assert out.numel() == n
        total_coltile = A.shape[1] // 32 + (1 if A.shape[1] % 32 != 0 else 0)
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                i = row * A.shape[1]
                j = col
                coltile = col // 32 + (1 if col % 32 != 0 else 0)
                rowtile = (row // 8 + (1 if row % 8 != 0 else 0)) * total_coltile
                offset = 32 * 8 * (rowtile + coltile)
                col2 = col % 32
                row2 = row % 8 * 32
                assert A.flatten()[i + j] == A[row, col]
    if orderOut == 'col32':
        out2, S = F.nvidia_transform(out, from_order=orderOut, to_order='row', state=S)
        torch.testing.assert_close(A, out2)