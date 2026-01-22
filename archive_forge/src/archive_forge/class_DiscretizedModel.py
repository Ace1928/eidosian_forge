import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class DiscretizedModel(GenericLikelihoodModel):
    """experimental model to fit discretized distribution

    Count models based on discretized distributions can be used to model
    data that is under- or over-dispersed relative to Poisson or that has
    heavier tails.

    Parameters
    ----------
    endog : array_like, 1-D
        Univariate data for fitting the distribution.
    exog : None
        Explanatory variables are not supported. The ``exog`` argument is
        only included for consistency in the signature across models.
    distr : DiscretizedCount instance
        (required) Instance of a DiscretizedCount distribution.

    See Also
    --------
    DiscretizedCount

    Examples
    --------
    >>> from scipy import stats
    >>> from statsmodels.distributions.discrete import (
            DiscretizedCount, DiscretizedModel)

    >>> dd = DiscretizedCount(stats.gamma)
    >>> mod = DiscretizedModel(y, distr=dd)
    >>> res = mod.fit()
    >>> probs = res.predict(which="probs", k_max=5)

    """

    def __init__(self, endog, exog=None, distr=None):
        if exog is not None:
            raise ValueError('exog is not supported')
        super().__init__(endog, exog, distr=distr)
        self._init_keys.append('distr')
        self.df_resid = len(endog) - distr.k_shapes
        self.df_model = 0
        self.k_extra = distr.k_shapes
        self.k_constant = 0
        self.nparams = distr.k_shapes
        self.start_params = 0.5 * np.ones(self.nparams)

    def loglike(self, params):
        args = params
        ll = np.log(self.distr._pmf(self.endog, *args))
        return ll.sum()

    def predict(self, params, exog=None, which=None, k_max=20):
        if exog is not None:
            raise ValueError('exog is not supported')
        args = params
        if which == 'probs':
            pr = self.distr.pmf(np.arange(k_max), *args)
            return pr
        else:
            raise ValueError('only which="probs" is currently implemented')

    def get_distr(self, params):
        """frozen distribution instance of the discrete distribution.
        """
        args = params
        distr = self.distr(*args)
        return distr