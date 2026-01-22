import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
class ConditionalMNLogit(_ConditionalModel):
    """
    Fit a conditional multinomial logit model to grouped data.

    Parameters
    ----------
    endog : array_like
        The dependent variable, must be integer-valued, coded
        0, 1, ..., c-1, where c is the number of response
        categories.
    exog : array_like
        The independent variables.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.

    Notes
    -----
    Equivalent to femlogit in Stata.

    References
    ----------
    Gary Chamberlain (1980).  Analysis of covariance with qualitative
    data. The Review of Economic Studies.  Vol. 47, No. 1, pp. 225-238.
    """

    def __init__(self, endog, exog, missing='none', **kwargs):
        super().__init__(endog, exog, missing=missing, **kwargs)
        self.endog = self.endog.astype(int)
        self.k_cat = self.endog.max() + 1
        self.df_model = (self.k_cat - 1) * self.exog.shape[1]
        self.df_resid = self.nobs - self.df_model
        self._ynames_map = {j: str(j) for j in range(self.k_cat)}
        self.J = self.k_cat
        self.K = self.exog.shape[1]
        if self.endog.min() < 0:
            msg = 'endog may not contain negative values'
            raise ValueError(msg)
        grx = collections.defaultdict(list)
        for k, v in enumerate(self.groups):
            grx[v].append(k)
        self._group_labels = list(grx.keys())
        self._group_labels.sort()
        self._grp_ix = [grx[k] for k in self._group_labels]

    def fit(self, start_params=None, method='BFGS', maxiter=100, full_output=True, disp=False, fargs=(), callback=None, retall=False, skip_hessian=False, **kwargs):
        if start_params is None:
            q = self.exog.shape[1]
            c = self.k_cat - 1
            start_params = np.random.normal(size=q * c)
        rslt = base.LikelihoodModel.fit(self, start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, skip_hessian=skip_hessian)
        rslt.params = rslt.params.reshape((self.exog.shape[1], -1))
        rslt = MultinomialResults(self, rslt)
        rslt.set_null_options(llnull=np.nan)
        return MultinomialResultsWrapper(rslt)

    def loglike(self, params):
        q = self.exog.shape[1]
        c = self.k_cat - 1
        pmat = params.reshape((q, c))
        pmat = np.concatenate((np.zeros((q, 1)), pmat), axis=1)
        lpr = np.dot(self.exog, pmat)
        ll = 0.0
        for ii in self._grp_ix:
            x = lpr[ii, :]
            jj = np.arange(x.shape[0], dtype=int)
            y = self.endog[ii]
            denom = 0.0
            for p in itertools.permutations(y):
                denom += np.exp(x[jj, p].sum())
            ll += x[jj, y].sum() - np.log(denom)
        return ll

    def score(self, params):
        q = self.exog.shape[1]
        c = self.k_cat - 1
        pmat = params.reshape((q, c))
        pmat = np.concatenate((np.zeros((q, 1)), pmat), axis=1)
        lpr = np.dot(self.exog, pmat)
        grad = np.zeros((q, c))
        for ii in self._grp_ix:
            x = lpr[ii, :]
            jj = np.arange(x.shape[0], dtype=int)
            y = self.endog[ii]
            denom = 0.0
            denomg = np.zeros((q, c))
            for p in itertools.permutations(y):
                v = np.exp(x[jj, p].sum())
                denom += v
                for i, r in enumerate(p):
                    if r != 0:
                        denomg[:, r - 1] += v * self.exog[ii[i], :]
            for i, r in enumerate(y):
                if r != 0:
                    grad[:, r - 1] += self.exog[ii[i], :]
            grad -= denomg / denom
        return grad.flatten()