import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
class BetaResults(GenericLikelihoodModelResults, _LLRMixin):
    """Results class for Beta regression

    This class inherits from GenericLikelihoodModelResults and not all
    inherited methods might be appropriate in this case.
    """

    @cache_readonly
    def fittedvalues(self):
        """In-sample predicted mean, conditional expectation."""
        return self.model.predict(self.params)

    @cache_readonly
    def fitted_precision(self):
        """In-sample predicted precision"""
        return self.model.predict(self.params, which='precision')

    @cache_readonly
    def resid(self):
        """Response residual"""
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def resid_pearson(self):
        """Pearson standardize residual"""
        std = np.sqrt(self.model.predict(self.params, which='var'))
        return self.resid / std

    @cache_readonly
    def prsquared(self):
        """Cox-Snell Likelihood-Ratio pseudo-R-squared.

        1 - exp((llnull - .llf) * (2 / nobs))
        """
        return self.pseudo_rsquared(kind='lr')

    def get_distribution_params(self, exog=None, exog_precision=None, transform=True):
        """
        Return distribution parameters converted from model prediction.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        (alpha, beta) : tuple of ndarrays
            Parameters for the scipy distribution to evaluate predictive
            distribution.
        """
        mean = self.predict(exog=exog, transform=transform)
        precision = self.predict(exog_precision=exog_precision, which='precision', transform=transform)
        return (precision * mean, precision * (1 - mean))

    def get_distribution(self, exog=None, exog_precision=None, transform=True):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        Instance of a scipy frozen distribution based on estimated
        parameters.

        See Also
        --------
        predict

        Notes
        -----
        This function delegates to the predict method to handle exog and
        exog_precision, which in turn makes any required transformations.

        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        from scipy import stats
        args = self.get_distribution_params(exog=exog, exog_precision=exog_precision, transform=transform)
        args = (np.asarray(arg) for arg in args)
        distr = stats.beta(*args)
        return distr

    def get_influence(self):
        """
        Get an instance of MLEInfluence with influence and outlier measures

        Returns
        -------
        infl : MLEInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        Support for mutli-link and multi-exog models is still experimental
        in MLEInfluence. Interface and some definitions might still change.

        Note: Difference to R betareg: Betareg has the same general leverage
        as this model. However, they use a linear approximation hat matrix
        to scale and studentize influence and residual statistics.
        MLEInfluence uses the generalized leverage as hat_matrix_diag.
        Additionally, MLEInfluence uses pearson residuals for residual
        analusis.

        References
        ----------
        todo

        """
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def bootstrap(self, *args, **kwargs):
        raise NotImplementedError