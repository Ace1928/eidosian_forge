import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old
def visit_fn(self, temperature, dim):
    """ Formula Visita from p. 405 of reference [2] """
    x, y = self.rand_gen.normal(size=(dim, 2)).T
    factor1 = np.exp(np.log(temperature) / (self._visiting_param - 1.0))
    factor4 = self._factor4_p * factor1
    x *= np.exp(-(self._visiting_param - 1.0) * np.log(self._factor6 / factor4) / (3.0 - self._visiting_param))
    den = np.exp((self._visiting_param - 1.0) * np.log(np.fabs(y)) / (3.0 - self._visiting_param))
    return x / den