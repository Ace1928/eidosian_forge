import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest

        Use
            geo_mean((x,y), (alpha, 1-alpha)) >= |z|
        as a reformulation of
            PowCone3D(x, y, z, alpha).

        Check validity of the reformulation by solving
        orthogonal projection problems.
        