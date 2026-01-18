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
@pytest.mark.parametrize('reduction', REDUCTIONS)
def test_reduce_empty_hidden(ops, dtype, reduction):
    func = getattr(ops, reduction)
    backprop_func = getattr(ops, f'backprop_{reduction}')
    lengths = ops.asarray1i([2, 3])
    Y = func(ops.alloc((5, 0), dtype=dtype), lengths)
    if reduction == 'reduce_max':
        Y, which = Y
        dX = backprop_func(Y, which, lengths)
    elif isinstance(Y, tuple):
        Y, extra = Y
        dX = backprop_func(Y, extra)
    else:
        dX = backprop_func(Y, lengths)
    assert Y.shape == (2, 0)
    assert dX.shape == (5, 0)