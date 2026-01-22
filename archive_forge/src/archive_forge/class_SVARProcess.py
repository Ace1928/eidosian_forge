import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
class SVARProcess(VARProcess):
    """
    Class represents a known SVAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    intercept : ndarray (length k)
    sigma_u : ndarray (k x k)
    names : sequence (length k)
    A : neqs x neqs np.ndarray with unknown parameters marked with 'E'
    A_mask : neqs x neqs mask array with known parameters masked
    B : neqs x neqs np.ndarry with unknown parameters marked with 'E'
    B_mask : neqs x neqs mask array with known parameters masked
    """

    def __init__(self, coefs, intercept, sigma_u, A_solve, B_solve, names=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.intercept = intercept
        self.sigma_u = sigma_u
        self.A_solve = A_solve
        self.B_solve = B_solve
        self.names = names

    def orth_ma_rep(self, maxn=10, P=None):
        """

        Unavailable for SVAR
        """
        raise NotImplementedError

    def svar_ma_rep(self, maxn=10, P=None):
        """

        Compute Structural MA coefficient matrices using MLE
        of A, B
        """
        if P is None:
            A_solve = self.A_solve
            B_solve = self.B_solve
            P = np.dot(npl.inv(A_solve), B_solve)
        ma_mats = self.ma_rep(maxn=maxn)
        return np.array([np.dot(coefs, P) for coefs in ma_mats])