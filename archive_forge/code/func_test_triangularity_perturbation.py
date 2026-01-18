import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_triangularity_perturbation(self):
    A = np.array([[0.32346, 30000.0, 30000.0, 30000.0], [0, 0.30089, 30000.0, 30000.0], [0, 0, 0.3221, 30000.0], [0, 0, 0, 0.30744]], dtype=float)
    A_logm = np.array([[-1.1286798202905046, 96141.83771420256, -4524855739.531793, 292496941103871.8], [0.0, -1.2010105295308229, 96346.96872113031, -4681048289.111054], [0.0, 0.0, -1.132893222644984, 95324.91830947757], [0.0, 0.0, 0.0, -1.1794753327255485]], dtype=float)
    assert_allclose(expm(A_logm), A, rtol=0.0001)
    random.seed(1234)
    tiny = 1e-17
    A_logm_perturbed = A_logm.copy()
    A_logm_perturbed[1, 0] = tiny
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Ill-conditioned.*')
        A_expm_logm_perturbed = expm(A_logm_perturbed)
    rtol = 0.0001
    atol = 100 * tiny
    assert_(not np.allclose(A_expm_logm_perturbed, A, rtol=rtol, atol=atol))