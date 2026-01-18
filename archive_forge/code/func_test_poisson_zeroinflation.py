import warnings
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.diagnostic_gen import (
from statsmodels.discrete._diagnostics_count import (
def test_poisson_zeroinflation(self, method='prob', exog_infl=None):
    """Test for excess zeros, zero inflation or deflation.

        Parameters
        ----------
        method : str
            Three methods ara available for the test:

             - "prob" : moment test for the probability of zeros
             - "broek" : score test against zero inflation with or without
                explanatory variables for inflation

        exog_infl : array_like or None
            Optional explanatory variables under the alternative of zero
            inflation, or deflation. Only used if method is "broek".

        Returns
        -------
        results

        Notes
        -----
        If method = "prob", then the moment test of He et al 1_ is used based
        on the explicit formula in Tang and Tang 2_.

        If method = "broek" and exog_infl is None, then the test by Van den
        Broek 3_ is used. This is a score test against and alternative of
        constant zero inflation or deflation.

        If method = "broek" and exog_infl is provided, then the extension of
        the broek test to varying zero inflation or deflation by Jansakul and
        Hinde is used.

        Warning: The Broek and the Jansakul and Hinde tests are not numerically
        stable when the probability of zeros in Poisson is small, i.e. if the
        conditional means of the estimated Poisson distribution are large.
        In these cases, p-values will not be accurate.
        """
    if method == 'prob':
        if exog_infl is not None:
            warnings.warn('exog_infl is only used if method = "broek"')
        res = test_poisson_zeros(self.results)
    elif method == 'broek':
        if exog_infl is None:
            res = test_poisson_zeroinflation_broek(self.results)
        else:
            exog_infl = np.asarray(exog_infl)
            if exog_infl.ndim == 1:
                exog_infl = exog_infl[:, None]
            res = test_poisson_zeroinflation_jh(self.results, exog_infl=exog_infl)
    return res