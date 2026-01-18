import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
@pytest.mark.parametrize('ops', XP_OPS)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_seq2col_lengths_zero_first_last(ops, dtype):
    cols_check = ops.asarray2f([[0, 0, 0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15, 0, 0, 0]], dtype=dtype)
    grad_check = ops.asarray2f([[2, 4, 6], [12, 15, 18], [21, 24, 27], [30, 33, 36], [26, 28, 30]], dtype=dtype)
    ops.xp.testing.assert_allclose(cols_check, ops.seq2col(ops.xp.arange(1.0, 16.0, dtype=dtype).reshape(5, 3), 1, lengths=ops.asarray1i([0, 5])))
    ops.xp.testing.assert_allclose(grad_check, ops.backprop_seq2col(cols_check, 1, lengths=ops.asarray1i([0, 5])))
    ops.xp.testing.assert_allclose(cols_check, ops.seq2col(ops.xp.arange(1.0, 16.0, dtype=dtype).reshape(5, 3), 1, lengths=ops.asarray1i([5, 0])))