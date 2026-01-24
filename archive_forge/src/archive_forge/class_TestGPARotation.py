import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
class TestGPARotation(unittest.TestCase):

    @staticmethod
    def str2matrix(A):
        A = A.lstrip().rstrip().split('\n')
        A = np.array([row.split() for row in A]).astype(float)
        return A

    @classmethod
    def get_A(cls):
        return cls.str2matrix('\n         .830 -.396\n         .818 -.469\n         .777 -.470\n         .798 -.401\n         .786  .500\n         .672  .458\n         .594  .444\n         .647  .333\n        ')

    @classmethod
    def get_quartimin_example(cls):
        A = cls.get_A()
        table_required = cls.str2matrix('\n          0.00000    0.42806   -0.46393    1.00000\n        1.00000    0.41311   -0.57313    0.25000\n        2.00000    0.38238   -0.36652    0.50000\n        3.00000    0.31850   -0.21011    0.50000\n        4.00000    0.20937   -0.13838    0.50000\n        5.00000    0.12379   -0.35583    0.25000\n        6.00000    0.04289   -0.53244    0.50000\n        7.00000    0.01098   -0.86649    0.50000\n        8.00000    0.00566   -1.65798    0.50000\n        9.00000    0.00558   -2.13212    0.25000\n       10.00000    0.00557   -2.49020    0.25000\n       11.00000    0.00557   -2.84585    0.25000\n       12.00000    0.00557   -3.20320    0.25000\n       13.00000    0.00557   -3.56143    0.25000\n       14.00000    0.00557   -3.92005    0.25000\n       15.00000    0.00557   -4.27885    0.25000\n       16.00000    0.00557   -4.63772    0.25000\n       17.00000    0.00557   -4.99663    0.25000\n       18.00000    0.00557   -5.35555    0.25000\n        ')
        L_required = cls.str2matrix('\n       0.891822   0.056015\n       0.953680  -0.023246\n       0.929150  -0.046503\n       0.876683   0.033658\n       0.013701   0.925000\n      -0.017265   0.821253\n      -0.052445   0.764953\n       0.085890   0.683115\n        ')
        return (A, table_required, L_required)

    @classmethod
    def get_biquartimin_example(cls):
        A = cls.get_A()
        table_required = cls.str2matrix('\n            0.00000    0.21632   -0.54955    1.00000\n            1.00000    0.19519   -0.46174    0.50000\n            2.00000    0.09479   -0.16365    1.00000\n            3.00000   -0.06302   -0.32096    0.50000\n            4.00000   -0.21304   -0.46562    1.00000\n            5.00000   -0.33199   -0.33287    1.00000\n            6.00000   -0.35108   -0.63990    0.12500\n            7.00000   -0.35543   -1.20916    0.12500\n            8.00000   -0.35568   -2.61213    0.12500\n            9.00000   -0.35568   -2.97910    0.06250\n           10.00000   -0.35568   -3.32645    0.06250\n           11.00000   -0.35568   -3.66021    0.06250\n           12.00000   -0.35568   -3.98564    0.06250\n           13.00000   -0.35568   -4.30635    0.06250\n           14.00000   -0.35568   -4.62451    0.06250\n           15.00000   -0.35568   -4.94133    0.06250\n           16.00000   -0.35568   -5.25745    0.06250\n        ')
        L_required = cls.str2matrix('\n           1.01753  -0.13657\n           1.11338  -0.24643\n           1.09200  -0.26890\n           1.00676  -0.16010\n          -0.26534   1.11371\n          -0.26972   0.99553\n          -0.29341   0.93561\n          -0.10806   0.80513\n        ')
        return (A, table_required, L_required)

    @classmethod
    def get_biquartimin_example_derivative_free(cls):
        A = cls.get_A()
        table_required = cls.str2matrix('\n            0.00000    0.21632   -0.54955    1.00000\n            1.00000    0.19519   -0.46174    0.50000\n            2.00000    0.09479   -0.16365    1.00000\n            3.00000   -0.06302   -0.32096    0.50000\n            4.00000   -0.21304   -0.46562    1.00000\n            5.00000   -0.33199   -0.33287    1.00000\n            6.00000   -0.35108   -0.63990    0.12500\n            7.00000   -0.35543   -1.20916    0.12500\n            8.00000   -0.35568   -2.61213    0.12500\n            9.00000   -0.35568   -2.97910    0.06250\n           10.00000   -0.35568   -3.32645    0.06250\n           11.00000   -0.35568   -3.66021    0.06250\n           12.00000   -0.35568   -3.98564    0.06250\n           13.00000   -0.35568   -4.30634    0.06250\n           14.00000   -0.35568   -4.62451    0.06250\n           15.00000   -0.35568   -4.94133    0.06250\n           16.00000   -0.35568   -6.32435    0.12500\n        ')
        L_required = cls.str2matrix('\n           1.01753  -0.13657\n           1.11338  -0.24643\n           1.09200  -0.26890\n           1.00676  -0.16010\n          -0.26534   1.11371\n          -0.26972   0.99553\n          -0.29342   0.93561\n          -0.10806   0.80513\n        ')
        return (A, table_required, L_required)

    @classmethod
    def get_quartimax_example_derivative_free(cls):
        A = cls.get_A()
        table_required = cls.str2matrix('\n        0.00000   -0.72073   -0.65498    1.00000\n        1.00000   -0.88561   -0.34614    2.00000\n        2.00000   -1.01992   -1.07152    1.00000\n        3.00000   -1.02237   -1.51373    0.50000\n        4.00000   -1.02269   -1.96205    0.50000\n        5.00000   -1.02273   -2.41116    0.50000\n        6.00000   -1.02273   -2.86037    0.50000\n        7.00000   -1.02273   -3.30959    0.50000\n        8.00000   -1.02273   -3.75881    0.50000\n        9.00000   -1.02273   -4.20804    0.50000\n       10.00000   -1.02273   -4.65726    0.50000\n       11.00000   -1.02273   -5.10648    0.50000\n        ')
        L_required = cls.str2matrix('\n       0.89876   0.19482\n       0.93394   0.12974\n       0.90213   0.10386\n       0.89281   0.17128\n       0.31558   0.87647\n       0.25113   0.77349\n       0.19801   0.71468\n       0.30786   0.65933\n        ')
        return (A, table_required, L_required)

    def test_orthomax(self):
        """
        Quartimax example
        http://www.stat.ucla.edu/research/gpa
        """
        A = self.get_A()
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=0, return_gradient=True)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        table_required = self.str2matrix('\n         0.00000   -0.72073   -0.65498    1.00000\n         1.00000   -0.88561   -0.34614    2.00000\n         2.00000   -1.01992   -1.07152    1.00000\n         3.00000   -1.02237   -1.51373    0.50000\n         4.00000   -1.02269   -1.96205    0.50000\n         5.00000   -1.02273   -2.41116    0.50000\n         6.00000   -1.02273   -2.86037    0.50000\n         7.00000   -1.02273   -3.30959    0.50000\n         8.00000   -1.02273   -3.75881    0.50000\n         9.00000   -1.02273   -4.20804    0.50000\n        10.00000   -1.02273   -4.65726    0.50000\n        11.00000   -1.02273   -5.10648    0.50000\n        ')
        L_required = self.str2matrix('\n        0.89876   0.19482\n        0.93394   0.12974\n        0.90213   0.10386\n        0.89281   0.17128\n        0.31558   0.87647\n        0.25113   0.77349\n        0.19801   0.71468\n        0.30786   0.65933\n        ')
        self.assertTrue(np.allclose(table, table_required, atol=1e-05))
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='orthogonal', return_gradient=True)
        L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L, L_oblimin, atol=1e-05))
        out = self.get_quartimax_example_derivative_free()
        A, table_required, L_required = out
        ff = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=0, return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(table, table_required, atol=1e-05))
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))

    def test_equivalence_orthomax_oblimin(self):
        """
        These criteria should be equivalent when restricted to orthogonal
        rotation.
        See Hartman 1976 page 299.
        """
        A = self.get_A()
        gamma = 0
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=True)
        L_orthomax, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma, rotation_method='orthogonal', return_gradient=True)
        L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L_orthomax, L_oblimin, atol=1e-05))
        gamma = 1
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=True)
        L_orthomax, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma, rotation_method='orthogonal', return_gradient=True)
        L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L_orthomax, L_oblimin, atol=1e-05))

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

    def test_oblimin(self):
        A, table_required, L_required = self.get_quartimin_example()
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='oblique')
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
        self.assertTrue(np.allclose(table, table_required, atol=1e-05))
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='oblique', return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='oblique')
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        self.assertTrue(np.allclose(table, table_required, atol=1e-05))
        A, table_required, L_required = self.get_biquartimin_example()
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=1 / 2, rotation_method='oblique')
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
        self.assertTrue(np.allclose(table, table_required, atol=1e-05))
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        out = self.get_biquartimin_example_derivative_free()
        A, table_required, L_required = out
        ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=1 / 2, rotation_method='oblique', return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='oblique')
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        self.assertTrue(np.allclose(table, table_required, atol=1e-05))

    def test_CF(self):
        out = self.get_quartimax_example_derivative_free()
        A, table_required, L_required = out
        vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=0, rotation_method='orthogonal', return_gradient=True)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=0, rotation_method='orthogonal', return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L, L_required, atol=1e-05))
        p, k = A.shape
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=1, return_gradient=True)
        L_vm, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=1 / p, rotation_method='orthogonal', return_gradient=True)
        L_CF, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T, kappa=1 / p, rotation_method='orthogonal', return_gradient=False)
        L_CF_df, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L_vm, L_CF, atol=1e-05))
        self.assertTrue(np.allclose(L_CF, L_CF_df, atol=1e-05))