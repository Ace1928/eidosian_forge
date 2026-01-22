import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def _get_sample_4d_4(self):
    points = [(0.0, 1.0)] * 4
    values = np.asarray([0.0, 1.0])
    values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
    values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
    values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
    values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
    values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
    return (points, values)