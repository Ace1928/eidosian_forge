import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('q', unit_quats)
def test_quaternion_reconstruction(q):
    M = nq.quat2mat(q)
    qt = nq.mat2quat(M)
    posm = np.allclose(q, qt)
    negm = np.allclose(q, -qt)
    assert posm or negm