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
@pytest.mark.parametrize('ops', CPU_OPS)
def test_seq2col_window_one_small(ops):
    seq = ops.asarray([[1.0], [3.0], [4.0], [5]], dtype='float32')
    cols = ops.seq2col(seq, 1)
    if hasattr(cols, 'get'):
        cols = cols.get()
    assert_allclose(cols[0], [0.0, 1.0, 3.0])
    assert_allclose(cols[1], [1.0, 3.0, 4.0])
    assert_allclose(cols[2], [3.0, 4.0, 5.0])
    assert_allclose(cols[3], [4.0, 5.0, 0.0])