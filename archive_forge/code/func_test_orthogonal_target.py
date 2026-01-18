import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def test_orthogonal_target(self):
    """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
    A = self.get_A()
    H = self.str2matrix('\n          .8 -.3\n          .8 -.4\n          .7 -.4\n          .9 -.4\n          .8  .5\n          .6  .4\n          .5  .4\n          .6  .3\n        ')
    vgQ = lambda L=None, A=None, T=None: vgQ_target(H, L=L, A=A, T=T)
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    table_required = self.str2matrix('\n        0.00000   0.05925  -0.61244   1.00000\n        1.00000   0.05444  -1.14701   0.12500\n        2.00000   0.05403  -1.68194   0.12500\n        3.00000   0.05399  -2.21689   0.12500\n        4.00000   0.05399  -2.75185   0.12500\n        5.00000   0.05399  -3.28681   0.12500\n        6.00000   0.05399  -3.82176   0.12500\n        7.00000   0.05399  -4.35672   0.12500\n        8.00000   0.05399  -4.89168   0.12500\n        9.00000   0.05399  -5.42664   0.12500\n        ')
    L_required = self.str2matrix('\n        0.84168  -0.37053\n        0.83191  -0.44386\n        0.79096  -0.44611\n        0.80985  -0.37650\n        0.77040   0.52371\n        0.65774   0.47826\n        0.58020   0.46189\n        0.63656   0.35255\n        ')
    self.assertTrue(np.allclose(table, table_required, atol=1e-05))
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    ff = lambda L=None, A=None, T=None: ff_target(H, L=L, A=A, T=T)
    L2, phi, T2, table = GPA(A, ff=ff, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L, L2, atol=1e-05))
    self.assertTrue(np.allclose(T, T2, atol=1e-05))
    vgQ = lambda L=None, A=None, T=None: vgQ_target(H, L=L, A=A, T=T, rotation_method='oblique')
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
    ff = lambda L=None, A=None, T=None: ff_target(H, L=L, A=A, T=T, rotation_method='oblique')
    L2, phi, T2, table = GPA(A, ff=ff, rotation_method='oblique')
    self.assertTrue(np.allclose(L, L2, atol=1e-05))
    self.assertTrue(np.allclose(T, T2, atol=1e-05))