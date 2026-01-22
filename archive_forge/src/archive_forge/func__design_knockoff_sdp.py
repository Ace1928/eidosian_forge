import numpy as np
import pandas as pd
from statsmodels.iolib import summary2
def _design_knockoff_sdp(exog):
    """
    Use semidefinite programming to construct a knockoff design
    matrix.

    Requires cvxopt to be installed.
    """
    try:
        from cvxopt import solvers, matrix
    except ImportError:
        raise ValueError('SDP knockoff designs require installation of cvxopt')
    nobs, nvar = exog.shape
    xnm = np.sum(exog ** 2, 0)
    xnm = np.sqrt(xnm)
    exog = exog / xnm
    Sigma = np.dot(exog.T, exog)
    c = matrix(-np.ones(nvar))
    h0 = np.concatenate((np.zeros(nvar), np.ones(nvar)))
    h0 = matrix(h0)
    G0 = np.concatenate((-np.eye(nvar), np.eye(nvar)), axis=0)
    G0 = matrix(G0)
    h1 = 2 * Sigma
    h1 = matrix(h1)
    i, j = np.diag_indices(nvar)
    G1 = np.zeros((nvar * nvar, nvar))
    G1[i * nvar + j, i] = 1
    G1 = matrix(G1)
    solvers.options['show_progress'] = False
    sol = solvers.sdp(c, G0, h0, [G1], [h1])
    sl = np.asarray(sol['x']).ravel()
    xcov = np.dot(exog.T, exog)
    exogn = _get_knmat(exog, xcov, sl)
    return (exog, exogn, sl)