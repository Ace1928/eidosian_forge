import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
class GaussianCovariance(ProcessCovariance):
    """
    An implementation of ProcessCovariance using the Gaussian kernel.

    This class represents a parametric covariance model for a Gaussian
    process as described in the work of Paciorek et al. cited below.

    Following Paciorek et al [1]_, the covariance between observations with
    index `i` and `j` is given by:

    .. math::

      s[i] \\cdot s[j] \\cdot h(|time[i] - time[j]| / \\sqrt{(u[i] + u[j]) /
      2}) \\cdot \\frac{u[i]^{1/4}u[j]^{1/4}}{\\sqrt{(u[i] + u[j])/2}}

    The ProcessMLE class allows linear models with this covariance
    structure to be fit using maximum likelihood (ML). The mean and
    covariance parameters of the model are fit jointly.

    The mean, scaling, and smoothing parameters can be linked to
    covariates.  The mean parameters are linked linearly, and the
    scaling and smoothing parameters use an log link function to
    preserve positivity.

    The reference of Paciorek et al. below provides more details.
    Note that here we only implement the 1-dimensional version of
    their approach.

    References
    ----------
    .. [1] Paciorek, C. J. and Schervish, M. J. (2006). Spatial modeling using
        a new class of nonstationary covariance functions. Environmetrics,
        17:483–506.
        https://papers.nips.cc/paper/2350-nonstationary-covariance-functions-for-gaussian-process-regression.pdf
    """

    def get_cov(self, time, sc, sm):
        da = np.subtract.outer(time, time)
        ds = np.add.outer(sm, sm) / 2
        qmat = da * da / ds
        cm = np.exp(-qmat / 2) / np.sqrt(ds)
        cm *= np.outer(sm, sm) ** 0.25
        cm *= np.outer(sc, sc)
        return cm

    def jac(self, time, sc, sm):
        da = np.subtract.outer(time, time)
        ds = np.add.outer(sm, sm) / 2
        sds = np.sqrt(ds)
        daa = da * da
        qmat = daa / ds
        p = len(time)
        eqm = np.exp(-qmat / 2)
        sm4 = np.outer(sm, sm) ** 0.25
        cmx = eqm * sm4 / sds
        dq0 = -daa / ds ** 2
        di = np.zeros((p, p))
        fi = np.zeros((p, p))
        scc = np.outer(sc, sc)
        jsm = []
        for i, _ in enumerate(sm):
            di *= 0
            di[i, :] += 0.5
            di[:, i] += 0.5
            dbottom = 0.5 * di / sds
            dtop = -0.5 * eqm * dq0 * di
            b = dtop / sds - eqm * dbottom / ds
            c = eqm / sds
            v = 0.25 * sm ** 0.25 / sm[i] ** 0.75
            fi *= 0
            fi[i, :] = v
            fi[:, i] = v
            fi[i, i] = 0.5 / sm[i] ** 0.5
            b = c * fi + b * sm4
            b *= scc
            jsm.append(b)
        jsc = []
        for i in range(0, len(sc)):
            b = np.zeros((p, p))
            b[i, :] = cmx[i, :] * sc
            b[:, i] += cmx[:, i] * sc
            jsc.append(b)
        return (jsc, jsm)