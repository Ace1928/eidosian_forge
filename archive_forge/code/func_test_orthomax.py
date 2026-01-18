import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def test_orthomax(self):
    """
        Quartimax example
        http://www.stat.ucla.edu/research/gpa
        """
    A = self.get_A()
    vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=0, return_gradient=True)
    L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    table_required = self.str2matrix('\n         0.00000   -0.72073   -0.65498    1.00000\n         1.00000   -0.88561   -0.34614    2.00000\n         2.00000   -1.01992   -1.07152    1.00000\n         3.00000   -1.02237   -1.51373    0.50000\n         4.00000   -1.02269   -1.96205    0.50000\n         5.00000   -1.02273   -2.41116    0.50000\n         6.00000   -1.02273   -2.86037    0.50000\n         7.00000   -1.02273   -3.30959    0.50000\n         8.00000   -1.02273   -3.75881    0.50000\n         9.00000   -1.02273   -4.20804    0.50000\n        10.00000   -1.02273   -4.65726    0.50000\n        11.00000   -1.02273   -5.10648    0.50000\n        ')
    L_required = self.str2matrix('\n        0.89876   0.19482\n        0.93394   0.12974\n        0.90213   0.10386\n        0.87651   0.17128\n        0.31558   0.87647\n        0.25113   0.77349\n        0.19801   0.71468\n        0.30786   0.65933\n        ')
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