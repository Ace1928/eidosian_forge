from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
@cache_readonly
def asqu(self):
    """Stephens 1974, does not have p-value formula for A^2"""
    nobs = self.nobs
    cdfvals = self.cdfvals
    asqu = -((2.0 * np.arange(1.0, nobs + 1) - 1) * (np.log(cdfvals) + np.log(1 - cdfvals[::-1]))).sum() / nobs - nobs
    return asqu