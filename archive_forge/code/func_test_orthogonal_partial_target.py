import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def test_orthogonal_partial_target(self):
    """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
    A = self.get_A()
    H = self.str2matrix('\n          .8 -.3\n          .8 -.4\n          .7 -.4\n          .9 -.4\n          .8  .5\n          .6  .4\n          .5  .4\n          .6  .3\n        ')
    W = self.str2matrix('\n        1 0\n        0 1\n        0 0\n        1 1\n        1 0\n        1 0\n        0 1\n        1 0\n        ')
    vgQ = lambda L=None, A=None, T=None: vgQ_partial_target(H, W, L=L, A=A, T=T)
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    table_required = self.str2matrix('\n         0.00000    0.02559   -0.84194    1.00000\n         1.00000    0.02203   -1.27116    0.25000\n         2.00000    0.02154   -1.71198    0.25000\n         3.00000    0.02148   -2.15713    0.25000\n         4.00000    0.02147   -2.60385    0.25000\n         5.00000    0.02147   -3.05114    0.25000\n         6.00000    0.02147   -3.49863    0.25000\n         7.00000    0.02147   -3.94619    0.25000\n         8.00000    0.02147   -4.39377    0.25000\n         9.00000    0.02147   -4.84137    0.25000\n        10.00000    0.02147   -5.28897    0.25000\n        ')
    L_required = self.str2matrix('\n        0.84526  -0.36228\n        0.83621  -0.43571\n        0.79528  -0.43836\n        0.81349  -0.36857\n        0.76525   0.53122\n        0.65303   0.48467\n        0.57565   0.46754\n        0.63308   0.35876\n        ')
    self.assertTrue(np.allclose(table, table_required, atol=1e-05))
    self.assertTrue(np.allclose(L, L_required, atol=1e-05))
    ff = lambda L=None, A=None, T=None: ff_partial_target(H, W, L=L, A=A, T=T)
    L2, phi, T2, table = GPA(A, ff=ff, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L, L2, atol=1e-05))
    self.assertTrue(np.allclose(T, T2, atol=1e-05))