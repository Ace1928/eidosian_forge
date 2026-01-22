from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
Initialize plane from its coefficients.

        Args:
            a: a coefficient of the plane.
            b: b coefficient of the plane.
            c: c coefficient of the plane.
            d: d coefficient of the plane.

        Returns:
            Plane.
        