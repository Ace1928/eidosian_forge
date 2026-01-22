import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
Compute interpolated values.

        Parameters
        ----------
        times : float or array_like
            Times of interest.
        order : {0, 1, 2}, optional
            Order of differentiation:

                * 0 (default) : return Rotation
                * 1 : return the angular rate in rad/sec
                * 2 : return the angular acceleration in rad/sec/sec

        Returns
        -------
        Interpolated Rotation, angular rate or acceleration.
        