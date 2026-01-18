import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_wtd_gradient_irls():
    np.random.seed(87342)
    fam = sm.families
    lnk = sm.families.links
    families = [(fam.Binomial, [lnk.Logit, lnk.Probit, lnk.CLogLog, lnk.Log, lnk.Cauchy]), (fam.Poisson, [lnk.Log, lnk.Identity, lnk.Sqrt]), (fam.Gamma, [lnk.Log, lnk.Identity, lnk.InversePower]), (fam.Gaussian, [lnk.Identity, lnk.Log, lnk.InversePower]), (fam.InverseGaussian, [lnk.Log, lnk.Identity, lnk.InversePower, lnk.InverseSquared]), (fam.NegativeBinomial, [lnk.Log, lnk.InversePower, lnk.InverseSquared, lnk.Identity])]
    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1
    skip_one = False
    for family_class, family_links in families:
        for link in family_links:
            for binom_version in (0, 1):
                method = 'bfgs'
                if family_class != fam.Binomial and binom_version == 1:
                    continue
                elif family_class == fam.Binomial and link == lnk.CLogLog:
                    continue
                elif family_class == fam.Binomial and link == lnk.Log:
                    continue
                elif (family_class, link) == (fam.Poisson, lnk.Identity):
                    lin_pred = 20 + exog.sum(1)
                elif (family_class, link) == (fam.Binomial, lnk.Log):
                    lin_pred = -1 + exog.sum(1) / 8
                elif (family_class, link) == (fam.Poisson, lnk.Sqrt):
                    lin_pred = -2 + exog.sum(1)
                elif (family_class, link) == (fam.Gamma, lnk.Log):
                    continue
                elif (family_class, link) == (fam.Gamma, lnk.Identity):
                    continue
                elif (family_class, link) == (fam.Gamma, lnk.InversePower):
                    continue
                elif (family_class, link) == (fam.Gaussian, lnk.Log):
                    continue
                elif (family_class, link) == (fam.Gaussian, lnk.InversePower):
                    continue
                elif (family_class, link) == (fam.InverseGaussian, lnk.Log):
                    lin_pred = -1 + exog.sum(1)
                    continue
                elif (family_class, link) == (fam.InverseGaussian, lnk.Identity):
                    lin_pred = 20 + 5 * exog.sum(1)
                    lin_pred = np.clip(lin_pred, 0.0001, np.inf)
                    continue
                elif (family_class, link) == (fam.InverseGaussian, lnk.InverseSquared):
                    lin_pred = 0.5 + exog.sum(1) / 5
                    continue
                elif (family_class, link) == (fam.InverseGaussian, lnk.InversePower):
                    lin_pred = 1 + exog.sum(1) / 5
                    method = 'newton'
                elif (family_class, link) == (fam.NegativeBinomial, lnk.Identity):
                    lin_pred = 20 + 5 * exog.sum(1)
                    lin_pred = np.clip(lin_pred, 0.001, np.inf)
                    method = 'newton'
                elif (family_class, link) == (fam.NegativeBinomial, lnk.InverseSquared):
                    lin_pred = 0.1 + np.random.uniform(size=exog.shape[0])
                    continue
                elif (family_class, link) == (fam.NegativeBinomial, lnk.InversePower):
                    lin_pred = 1 + exog.sum(1) / 5
                    continue
                elif (family_class, link) == (fam.Gaussian, lnk.InversePower):
                    skip_one = True
                else:
                    lin_pred = np.random.uniform(size=exog.shape[0])
                endog = gen_endog(lin_pred, family_class, link, binom_version)
                if binom_version == 0:
                    wts = np.ones_like(endog)
                    tmp = np.random.randint(2, 5, size=(endog > endog.mean()).sum())
                    wts[endog > endog.mean()] = tmp
                else:
                    wts = np.ones(shape=endog.shape[0])
                    y = endog[:, 0] / endog.sum(axis=1)
                    tmp = np.random.gamma(2, size=(y > y.mean()).sum())
                    wts[y > y.mean()] = tmp
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mod_irls = sm.GLM(endog, exog, var_weights=wts, family=family_class(link=link()))
                rslt_irls = mod_irls.fit(method='IRLS', atol=1e-10, tol_criterion='params')
                for max_start_irls, start_params in ((0, rslt_irls.params), (3, None)):
                    if max_start_irls > 0 and skip_one:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mod_gradient = sm.GLM(endog, exog, var_weights=wts, family=family_class(link=link()))
                    rslt_gradient = mod_gradient.fit(max_start_irls=max_start_irls, start_params=start_params, method=method)
                    assert_allclose(rslt_gradient.params, rslt_irls.params, rtol=1e-06, atol=5e-05)
                    assert_allclose(rslt_gradient.llf, rslt_irls.llf, rtol=1e-06, atol=1e-06)
                    assert_allclose(rslt_gradient.scale, rslt_irls.scale, rtol=1e-06, atol=1e-06)
                    gradient_bse = rslt_gradient.bse
                    ehess = mod_gradient.hessian(rslt_gradient.params, observed=False)
                    gradient_bse = np.sqrt(-np.diag(np.linalg.inv(ehess)))
                    assert_allclose(gradient_bse, rslt_irls.bse, rtol=1e-06, atol=5e-05)