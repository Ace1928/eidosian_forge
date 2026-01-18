from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def vectorsToMatrix(aa, bb):
    """
    Performs the vector multiplication of the elements of two vectors, constructing the 3x3 matrix.

    Args:
        aa: One vector of size 3
        bb: Another vector of size 3

    Returns:
        A 3x3 matrix M composed of the products of the elements of aa and bb : M_ij = aa_i * bb_j.
    """
    MM = np.zeros([3, 3], float)
    for ii in range(3):
        for jj in range(3):
            MM[ii, jj] = aa[ii] * bb[jj]
    return MM