import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def _est_cond_mean(self):
    """
        Calculates the expected conditional mean
        m(X, Z=l) for all possible l
        """
    self.dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))
    X = copy.deepcopy(self.exog)
    m = 0
    for i in self.dom_x:
        X[:, self.test_vars] = i
        m += self.model.fit(data_predict=X)[0]
    m = m / float(len(self.dom_x))
    m = np.reshape(m, (np.shape(self.exog)[0], 1))
    return m