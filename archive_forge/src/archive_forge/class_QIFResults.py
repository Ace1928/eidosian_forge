import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
class QIFResults(base.LikelihoodModelResults):
    """Results class for QIF Regression"""

    def __init__(self, model, params, cov_params, scale, use_t=False, **kwds):
        super().__init__(model, params, normalized_cov_params=cov_params, scale=scale)
        self.qif, _, _, _, _ = self.model.objective(params)

    @cache_readonly
    def aic(self):
        """
        An AIC-like statistic for models fit using QIF.
        """
        if isinstance(self.model.cov_struct, QIFIndependence):
            msg = 'AIC not available with QIFIndependence covariance'
            raise ValueError(msg)
        df = self.model.exog.shape[1]
        return self.qif + 2 * df

    @cache_readonly
    def bic(self):
        """
        A BIC-like statistic for models fit using QIF.
        """
        if isinstance(self.model.cov_struct, QIFIndependence):
            msg = 'BIC not available with QIFIndependence covariance'
            raise ValueError(msg)
        df = self.model.exog.shape[1]
        return self.qif + np.log(self.model.nobs) * df

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values from the model.
        """
        return self.model.family.link.inverse(np.dot(self.model.exog, self.params))

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the QIF regression results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        top_left = [('Dep. Variable:', None), ('Method:', ['QIF']), ('Family:', [self.model.family.__class__.__name__]), ('Covariance structure:', [self.model.cov_struct.__class__.__name__]), ('Date:', None), ('Time:', None)]
        NY = [len(y) for y in self.model.groups_ix]
        top_right = [('No. Observations:', [sum(NY)]), ('No. clusters:', [len(NY)]), ('Min. cluster size:', [min(NY)]), ('Max. cluster size:', [max(NY)]), ('Mean cluster size:', ['%.1f' % np.mean(NY)]), ('Scale:', ['%.3f' % self.scale])]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        if xname is None:
            xname = self.model.exog_names
        if yname is None:
            yname = self.model.endog_names
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=False)
        return smry