import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def _compute_J(self, A_solve, B_solve):
    neqs = self.neqs
    sigma_u = self.sigma_u
    A_mask = self.A_mask
    B_mask = self.B_mask
    D_nT = np.zeros([int(1.0 / 2 * neqs * (neqs + 1)), neqs ** 2])
    for j in range(neqs):
        i = j
        while j <= i < neqs:
            u = np.zeros([int(1.0 / 2 * neqs * (neqs + 1)), 1])
            u[int(j * neqs + (i + 1) - 1.0 / 2 * (j + 1) * j - 1)] = 1
            Tij = np.zeros([neqs, neqs])
            Tij[i, j] = 1
            Tij[j, i] = 1
            D_nT = D_nT + np.dot(u, Tij.ravel('F')[:, None].T)
            i = i + 1
    D_n = D_nT.T
    D_pl = npl.pinv(D_n)
    S_B = np.zeros((neqs ** 2, len(A_solve[A_mask])))
    S_D = np.zeros((neqs ** 2, len(B_solve[B_mask])))
    j = 0
    j_d = 0
    if len(A_solve[A_mask]) != 0:
        A_vec = np.ravel(A_mask, order='F')
        for k in range(neqs ** 2):
            if A_vec[k]:
                S_B[k, j] = -1
                j += 1
    if len(B_solve[B_mask]) != 0:
        B_vec = np.ravel(B_mask, order='F')
        for k in range(neqs ** 2):
            if B_vec[k]:
                S_D[k, j_d] = 1
                j_d += 1
    invA = npl.inv(A_solve)
    J_p1i = np.dot(np.dot(D_pl, np.kron(sigma_u, invA)), S_B)
    J_p1 = -2.0 * J_p1i
    J_p2 = np.dot(np.dot(D_pl, np.kron(invA, invA)), S_D)
    J = np.append(J_p1, J_p2, axis=1)
    return J