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
def test_use_ops():
    class_ops = get_current_ops()
    with use_ops('numpy'):
        new_ops = get_current_ops()
        assert new_ops.name == 'numpy'
    with use_ops('cupy'):
        new_ops = get_current_ops()
        assert new_ops.name == 'cupy'
    new_ops = get_current_ops()
    assert class_ops.name == new_ops.name