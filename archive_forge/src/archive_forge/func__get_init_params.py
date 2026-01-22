import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def _get_init_params(self, A_guess, B_guess):
    """
        Returns either the given starting or .1 if none are given.
        """
    var_type = self.svar_type.lower()
    n_masked_a = self.A_mask.sum()
    if var_type in ['ab', 'a']:
        if A_guess is None:
            A_guess = np.array([0.1] * n_masked_a)
        elif len(A_guess) != n_masked_a:
            msg = 'len(A_guess) = %s, there are %s parameters in A'
            raise ValueError(msg % (len(A_guess), n_masked_a))
    else:
        A_guess = []
    n_masked_b = self.B_mask.sum()
    if var_type in ['ab', 'b']:
        if B_guess is None:
            B_guess = np.array([0.1] * n_masked_b)
        elif len(B_guess) != n_masked_b:
            msg = 'len(B_guess) = %s, there are %s parameters in B'
            raise ValueError(msg % (len(B_guess), n_masked_b))
    else:
        B_guess = []
    return np.r_[A_guess, B_guess]