import numpy as np
class UnivariateFunc1(_UnivariateFunction):
    """

    made up, with sin and quadratic trend
    """

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        if x is None and distr_x is None:
            from scipy import stats
            distr_x = stats.uniform(-2, 4)
        else:
            nobs = x.shape[0]
        self.s_noise = 2.0
        self.func = func1
        super().__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)

    def het_scale(self, x):
        return np.sqrt(np.abs(3 + x))