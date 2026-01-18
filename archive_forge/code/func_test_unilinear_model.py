import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_unilinear_model(self):
    x = np.linspace(0.0, 5.0)
    y = 1.0 * x + 2.0
    data = Data(x, y)
    odr_obj = ODR(data, unilinear)
    output = odr_obj.run()
    assert_array_almost_equal(output.beta, [1.0, 2.0])