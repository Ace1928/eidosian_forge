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
def test_seq2col_window_two(ops, dtype):
    seq = ops.asarray([[1.0], [2.0], [3.0], [4]], dtype=dtype)
    cols = ops.seq2col(seq, 2)
    if not isinstance(cols, numpy.ndarray):
        cols = cols.get()
    assert_allclose(cols[0], [0.0, 0.0, 1.0, 2.0, 3.0])
    assert_allclose(cols[1], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert_allclose(cols[2], [1.0, 2.0, 3.0, 4.0, 0.0])
    assert_allclose(cols[3], [2.0, 3.0, 4.0, 0.0, 0.0])