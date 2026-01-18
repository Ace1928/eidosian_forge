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
@pytest.mark.parametrize('ops', ALL_OPS)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_reduce_sum(ops, dtype):
    X = ops.asarray2f([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    lengths = ops.asarray1i([3, 2])
    ops.xp.testing.assert_allclose(ops.reduce_sum(X, lengths), [[9.0, 12.0], [4.0, 6.0]])
    lengths = ops.asarray1i([3, 0, 2])
    ops.xp.testing.assert_allclose(ops.reduce_sum(X, lengths), [[9.0, 12.0], [0.0, 0.0], [4.0, 6.0]])
    with pytest.raises(IndexError):
        ops.reduce_sum(X, ops.xp.array([5, 5, 5, 5], dtype='i'))
    with pytest.raises(ValueError):
        ops.reduce_sum(X, ops.xp.array([-1, 10, 5, 5], dtype='i'))