import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_quat2mat():
    M = nq.quat2mat([1, 0, 0, 0])
    (assert_array_almost_equal, M, np.eye(3))
    M = nq.quat2mat([3, 0, 0, 0])
    (assert_array_almost_equal, M, np.eye(3))
    M = nq.quat2mat([0, 1, 0, 0])
    (assert_array_almost_equal, M, np.diag([1, -1, -1]))
    M = nq.quat2mat([0, 2, 0, 0])
    (assert_array_almost_equal, M, np.diag([1, -1, -1]))
    M = nq.quat2mat([0, 0, 0, 0])
    (assert_array_almost_equal, M, np.eye(3))