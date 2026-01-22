import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
class QuantRegResults(RegressionResults):
    """Results instance for the QuantReg model"""

    @cache_readonly
    def prsquared(self):
        q = self.q
        endog = self.model.endog
        e = self.resid
        e = np.where(e < 0, (1 - q) * e, q * e)
        e = np.abs(e)
        ered = endog - stats.scoreatpercentile(endog, q * 100)
        ered = np.where(ered < 0, (1 - q) * ered, q * ered)
        ered = np.abs(ered)
        return 1 - np.sum(e) / np.sum(ered)

    def scale(self):
        return 1.0

    @cache_readonly
    def bic(self):
        return np.nan

    @cache_readonly
    def aic(self):
        return np.nan

    @cache_readonly
    def llf(self):
        return np.nan

    @cache_readonly
    def rsquared(self):
        return np.nan

    @cache_readonly
    def rsquared_adj(self):
        return np.nan

    @cache_readonly
    def mse(self):
        return np.nan

    @cache_readonly
    def mse_model(self):
        return np.nan

    @cache_readonly
    def mse_total(self):
        return np.nan

    @cache_readonly
    def centered_tss(self):
        return np.nan

    @cache_readonly
    def uncentered_tss(self):
        return np.nan

    @cache_readonly
    def HC0_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC1_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC2_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC3_se(self):
        raise NotImplementedError

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        eigvals = self.eigenvals
        condno = self.condition_number
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['Least Squares']), ('Date:', None), ('Time:', None)]
        top_right = [('Pseudo R-squared:', ['%#8.4g' % self.prsquared]), ('Bandwidth:', ['%#8.4g' % self.bandwidth]), ('Sparsity:', ['%#8.4g' % self.sparsity]), ('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None)]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        etext = []
        if eigvals[-1] < 1e-10:
            wstr = 'The smallest eigenvalue is %6.3g. This might indicate '
            wstr += 'that there are\n'
            wstr += 'strong multicollinearity problems or that the design '
            wstr += 'matrix is singular.'
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:
            wstr = 'The condition number is large, %6.3g. This might '
            wstr += 'indicate that there are\n'
            wstr += 'strong multicollinearity or other numerical '
            wstr += 'problems.'
            wstr = wstr % condno
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry