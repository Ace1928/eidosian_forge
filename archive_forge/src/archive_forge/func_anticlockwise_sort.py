from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def anticlockwise_sort(pps):
    """
    Sort a list of 2D points in anticlockwise order

    Args:
        pps: List of points to be sorted

    Returns:
        Sorted list of points.
    """
    new_pps = []
    angles = np.zeros(len(pps), float)
    for ipp, pp in enumerate(pps):
        angles[ipp] = np.arctan2(pp[1], pp[0])
    idx_sorted = np.argsort(angles)
    for ii in range(len(pps)):
        new_pps.append(pps[idx_sorted[ii]])
    return new_pps