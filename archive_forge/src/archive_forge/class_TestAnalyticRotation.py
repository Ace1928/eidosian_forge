import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
class TestAnalyticRotation(unittest.TestCase):

    @staticmethod
    def str2matrix(A):
        A = A.lstrip().rstrip().split('\n')
        A = np.array([row.split() for row in A]).astype(float)
        return A

    def test_target_rotation(self):
        """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
        A = self.str2matrix('\n         .830 -.396\n         .818 -.469\n         .777 -.470\n         .798 -.401\n         .786  .500\n         .672  .458\n         .594  .444\n         .647  .333\n        ')
        H = self.str2matrix('\n          .8 -.3\n          .8 -.4\n          .7 -.4\n          .9 -.4\n          .8  .5\n          .6  .4\n          .5  .4\n          .6  .3\n        ')
        T = target_rotation(A, H)
        L = A.dot(T)
        L_required = self.str2matrix('\n        0.84168  -0.37053\n        0.83191  -0.44386\n        0.79096  -0.44611\n        0.80985  -0.37650\n        0.77040   0.52371\n        0.65774   0.47826\n        0.58020   0.46189\n        0.63656   0.35255\n        ')
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        T = target_rotation(A, H, full_rank=True)
        L = A.dot(T)
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))

    def test_orthogonal_target(self):
        """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
        A = self.str2matrix('\n         .830 -.396\n         .818 -.469\n         .777 -.470\n         .798 -.401\n         .786  .500\n         .672  .458\n         .594  .444\n         .647  .333\n        ')
        H = self.str2matrix('\n          .8 -.3\n          .8 -.4\n          .7 -.4\n          .9 -.4\n          .8  .5\n          .6  .4\n          .5  .4\n          .6  .3\n        ')
        vgQ = lambda L=None, A=None, T=None: vgQ_target(H, L=L, A=A, T=T)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        T_analytic = target_rotation(A, H)
        self.assertTrue(np.allclose(T, T_analytic, atol=1e-05))