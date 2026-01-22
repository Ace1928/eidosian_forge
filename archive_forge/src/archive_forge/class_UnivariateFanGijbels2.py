import numpy as np
class UnivariateFanGijbels2(_UnivariateFunction):
    __doc__ = _UnivariateFunction.__doc__ % doc

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.0
        self.s_noise = 0.5
        self.func = fg2
        super(self.__class__, self).__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)