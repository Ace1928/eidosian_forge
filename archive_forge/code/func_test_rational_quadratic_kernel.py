from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def test_rational_quadratic_kernel():
    kernel = RationalQuadratic(length_scale=[1.0, 1.0])
    message = 'RationalQuadratic kernel only supports isotropic version, please use a single scalar for length_scale'
    with pytest.raises(AttributeError, match=message):
        kernel(X)