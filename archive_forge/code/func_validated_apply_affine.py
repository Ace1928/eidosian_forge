from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ..affines import (
from ..eulerangles import euler2mat
from ..orientations import aff2axcodes
def validated_apply_affine(T, xyz):
    xyz = np.asarray(xyz)
    shape = xyz.shape[0:-1]
    XYZ = np.dot(np.reshape(xyz, (np.prod(shape), 3)), T[0:3, 0:3].T)
    XYZ[:, 0] += T[0, 3]
    XYZ[:, 1] += T[1, 3]
    XYZ[:, 2] += T[2, 3]
    XYZ = np.reshape(XYZ, shape + (3,))
    return XYZ