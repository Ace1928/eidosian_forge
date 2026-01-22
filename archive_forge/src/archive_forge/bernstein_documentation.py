import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
Generate random numbers from distribution.

        Parameters
        ----------
        nobs : int
            Number of random observations to generate.
        