import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
class _ConditionalModel(base.LikelihoodModel):

    def __init__(self, endog, exog, missing='none', **kwargs):
        if 'groups' not in kwargs:
            raise ValueError("'groups' is a required argument")
        groups = kwargs['groups']
        if groups.size != endog.size:
            msg = "'endog' and 'groups' should have the same dimensions"
            raise ValueError(msg)
        if exog.shape[0] != endog.size:
            msg = "The leading dimension of 'exog' should equal the length of 'endog'"
            raise ValueError(msg)
        super().__init__(endog, exog, missing=missing, **kwargs)
        if self.data.const_idx is not None:
            msg = 'Conditional models should not have an intercept in the ' + 'design matrix'
            raise ValueError(msg)
        exog = self.exog
        self.k_params = exog.shape[1]
        row_ix = {}
        for i, g in enumerate(groups):
            if g not in row_ix:
                row_ix[g] = []
            row_ix[g].append(i)
        endog, exog = (np.asarray(endog), np.asarray(exog))
        offset = kwargs.get('offset')
        self._endog_grp = []
        self._exog_grp = []
        self._groupsize = []
        if offset is not None:
            offset = np.asarray(offset)
            self._offset_grp = []
        self._offset = []
        self._sumy = []
        self.nobs = 0
        drops = [0, 0]
        for g, ix in row_ix.items():
            y = endog[ix].flat
            if np.std(y) == 0:
                drops[0] += 1
                drops[1] += len(y)
                continue
            self.nobs += len(y)
            self._endog_grp.append(y)
            if offset is not None:
                self._offset_grp.append(offset[ix])
            self._groupsize.append(len(y))
            self._exog_grp.append(exog[ix, :])
            self._sumy.append(np.sum(y))
        if drops[0] > 0:
            msg = ('Dropped %d groups and %d observations for having ' + 'no within-group variance') % tuple(drops)
            warnings.warn(msg)
        if offset is not None:
            self._endofs = []
            for k, ofs in enumerate(self._offset_grp):
                self._endofs.append(np.dot(self._endog_grp[k], ofs))
        self._n_groups = len(self._endog_grp)
        self._xy = []
        self._n1 = []
        for g in range(self._n_groups):
            self._xy.append(np.dot(self._endog_grp[g], self._exog_grp[g]))
            self._n1.append(np.sum(self._endog_grp[g]))

    def hessian(self, params):
        from statsmodels.tools.numdiff import approx_fprime
        hess = approx_fprime(params, self.score)
        hess = np.atleast_2d(hess)
        return hess

    def fit(self, start_params=None, method='BFGS', maxiter=100, full_output=True, disp=False, fargs=(), callback=None, retall=False, skip_hessian=False, **kwargs):
        rslt = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, skip_hessian=skip_hessian)
        crslt = ConditionalResults(self, rslt.params, rslt.cov_params(), 1)
        crslt.method = method
        crslt.nobs = self.nobs
        crslt.n_groups = self._n_groups
        crslt._group_stats = ['%d' % min(self._groupsize), '%d' % max(self._groupsize), '%.1f' % np.mean(self._groupsize)]
        rslt = ConditionalResultsWrapper(crslt)
        return rslt

    def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, **kwargs):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword argument that are used when fitting the model.

        Returns
        -------
        Results
            A results instance.
        """
        from statsmodels.base.elastic_net import fit_elasticnet
        if method != 'elastic_net':
            raise ValueError('method for fit_regularized must be elastic_net')
        defaults = {'maxiter': 50, 'L1_wt': 1, 'cnvrg_tol': 1e-10, 'zero_tol': 1e-10}
        defaults.update(kwargs)
        return fit_elasticnet(self, method=method, alpha=alpha, start_params=start_params, refit=refit, **defaults)

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        try:
            groups = kwargs['groups']
            del kwargs['groups']
        except KeyError:
            raise ValueError("'groups' is a required argument")
        if isinstance(groups, str):
            groups = data[groups]
        if '0+' not in formula.replace(' ', ''):
            warnings.warn('Conditional models should not include an intercept')
        model = super().from_formula(formula, *args, data=data, groups=groups, **kwargs)
        return model