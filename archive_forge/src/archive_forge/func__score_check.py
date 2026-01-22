import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def _score_check(self, params):
    """Inherited score with finite differences

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score based on numerical derivatives
        """
    return super().score(params)