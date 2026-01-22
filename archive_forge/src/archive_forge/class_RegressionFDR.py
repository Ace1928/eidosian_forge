import numpy as np
import pandas as pd
from statsmodels.iolib import summary2
class RegressionFDR:
    """
    Control FDR in a regression procedure.

    Parameters
    ----------
    endog : array_like
        The dependent variable of the regression
    exog : array_like
        The independent variables of the regression
    regeffects : RegressionEffects instance
        An instance of a RegressionEffects class that can compute
        effect sizes for the regression coefficients.
    method : str
        The approach used to assess and control FDR, currently
        must be 'knockoff'.

    Returns
    -------
    Returns an instance of the RegressionFDR class.  The `fdr` attribute
    holds the estimated false discovery rates.

    Notes
    -----
    This class Implements the knockoff method of Barber and Candes.
    This is an approach for controlling the FDR of a variety of
    regression estimation procedures, including correlation
    coefficients, OLS regression, OLS with forward selection, and
    LASSO regression.

    For other approaches to FDR control in regression, see the
    statsmodels.stats.multitest module.  Methods provided in that
    module use Z-scores or p-values, and therefore require standard
    errors for the coefficient estimates to be available.

    The default method for constructing the augmented design matrix is
    the 'equivariant' approach, set `design_method='sdp'` to use an
    alternative approach involving semidefinite programming.  See
    Barber and Candes for more information about both approaches.  The
    sdp approach requires that the cvxopt package be installed.
    """

    def __init__(self, endog, exog, regeffects, method='knockoff', **kwargs):
        if hasattr(exog, 'columns'):
            self.xnames = exog.columns
        else:
            self.xnames = ['x%d' % j for j in range(exog.shape[1])]
        exog = np.asarray(exog)
        endog = np.asarray(endog)
        if 'design_method' not in kwargs:
            kwargs['design_method'] = 'equi'
        nobs, nvar = exog.shape
        if kwargs['design_method'] == 'equi':
            exog1, exog2, _ = _design_knockoff_equi(exog)
        elif kwargs['design_method'] == 'sdp':
            exog1, exog2, _ = _design_knockoff_sdp(exog)
        endog = endog - np.mean(endog)
        self.endog = endog
        self.exog = np.concatenate((exog1, exog2), axis=1)
        self.exog1 = exog1
        self.exog2 = exog2
        self.stats = regeffects.stats(self)
        unq, inv, cnt = np.unique(self.stats, return_inverse=True, return_counts=True)
        cc = np.cumsum(cnt)
        denom = len(self.stats) - cc + cnt
        denom[denom < 1] = 1
        ii = np.searchsorted(unq, -unq, side='right') - 1
        numer = cc[ii]
        numer[ii < 0] = 0
        fdrp = (1 + numer) / denom
        fdr = numer / denom
        self.fdr = fdr[inv]
        self.fdrp = fdrp[inv]
        self._ufdr = fdr
        self._unq = unq
        df = pd.DataFrame(index=self.xnames)
        df['Stat'] = self.stats
        df['FDR+'] = self.fdrp
        df['FDR'] = self.fdr
        self.fdr_df = df

    def threshold(self, tfdr):
        """
        Returns the threshold statistic for a given target FDR.
        """
        if np.min(self._ufdr) <= tfdr:
            return self._unq[self._ufdr <= tfdr][0]
        else:
            return np.inf

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('Regression FDR results')
        summ.add_df(self.fdr_df)
        return summ