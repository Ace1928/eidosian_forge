import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
def _inertia_eigvals_to_axes_lengths_3D(inertia_tensor_eigvals):
    """Compute ellipsoid axis lengths from inertia tensor eigenvalues.

    Parameters
    ---------
    inertia_tensor_eigvals : sequence of float
        A sequence of 3 floating point eigenvalues, sorted in descending order.

    Returns
    -------
    axis_lengths : list of float
        The ellipsoid axis lengths sorted in descending order.

    Notes
    -----
    Let a >= b >= c be the ellipsoid semi-axes and s1 >= s2 >= s3 be the
    inertia tensor eigenvalues.

    The inertia tensor eigenvalues are given for a solid ellipsoid in [1]_.
    s1 = 1 / 5 * (a**2 + b**2)
    s2 = 1 / 5 * (a**2 + c**2)
    s3 = 1 / 5 * (b**2 + c**2)

    Rearranging to solve for a, b, c in terms of s1, s2, s3 gives
    a = math.sqrt(5 / 2 * ( s1 + s2 - s3))
    b = math.sqrt(5 / 2 * ( s1 - s2 + s3))
    c = math.sqrt(5 / 2 * (-s1 + s2 + s3))

    We can then simply replace sqrt(5/2) by sqrt(10) to get the full axes
    lengths rather than the semi-axes lengths.

    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors  # noqa
    """
    axis_lengths = []
    for ax in range(2, -1, -1):
        w = sum((v * -1 if i == ax else v for i, v in enumerate(inertia_tensor_eigvals)))
        axis_lengths.append(sqrt(10 * w))
    return axis_lengths