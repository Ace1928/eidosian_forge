from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def vb_elbo_grad_base(self, h, tm, tv, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd):
    """
        Return the gradient of the ELBO function.

        See vb_elbo_base for parameters.
        """
    fep_mean_grad = 0.0
    fep_sd_grad = 0.0
    vcp_mean_grad = 0.0
    vcp_sd_grad = 0.0
    vc_mean_grad = 0.0
    vc_sd_grad = 0.0
    for w in glw:
        z = self.rng * w[1]
        u = h(z) * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        r = u / np.sqrt(tv)
        fep_mean_grad += w[0] * np.dot(u, self.exog)
        vc_mean_grad += w[0] * self.exog_vc.transpose().dot(u)
        fep_sd_grad += w[0] * z * np.dot(r, self.exog ** 2 * fep_sd)
        v = self.exog_vc2.multiply(vc_sd).transpose().dot(r)
        v = np.squeeze(np.asarray(v))
        vc_sd_grad += w[0] * z * v
    fep_mean_grad *= self.rng
    vc_mean_grad *= self.rng
    fep_sd_grad *= self.rng
    vc_sd_grad *= self.rng
    fep_mean_grad += np.dot(self.endog, self.exog)
    vc_mean_grad += self.exog_vc.transpose().dot(self.endog)
    fep_mean_grad_i, fep_sd_grad_i, vcp_mean_grad_i, vcp_sd_grad_i, vc_mean_grad_i, vc_sd_grad_i = self._elbo_grad_common(fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd)
    fep_mean_grad += fep_mean_grad_i
    fep_sd_grad += fep_sd_grad_i
    vcp_mean_grad += vcp_mean_grad_i
    vcp_sd_grad += vcp_sd_grad_i
    vc_mean_grad += vc_mean_grad_i
    vc_sd_grad += vc_sd_grad_i
    fep_sd_grad += 1 / fep_sd
    vcp_sd_grad += 1 / vcp_sd
    vc_sd_grad += 1 / vc_sd
    mean_grad = np.concatenate((fep_mean_grad, vcp_mean_grad, vc_mean_grad))
    sd_grad = np.concatenate((fep_sd_grad, vcp_sd_grad, vc_sd_grad))
    if self.verbose:
        print('|G|=%f' % np.sqrt(np.sum(mean_grad ** 2) + np.sum(sd_grad ** 2)))
    return (mean_grad, sd_grad)