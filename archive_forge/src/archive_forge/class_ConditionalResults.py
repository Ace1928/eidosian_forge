import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
import collections
import warnings
import itertools
class ConditionalResults(base.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params, scale):
        super().__init__(model, params, normalized_cov_params=normalized_cov_params, scale=scale)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the fitted model.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Log-Likelihood:', None), ('Method:', [self.method]), ('Date:', None), ('Time:', None)]
        top_right = [('No. Observations:', None), ('No. groups:', [self.n_groups]), ('Min group size:', [self._group_stats[0]]), ('Max group size:', [self._group_stats[1]]), ('Mean group size:', [self._group_stats[2]])]
        if title is None:
            title = 'Conditional Logit Model Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        return smry