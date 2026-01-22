import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
@np.vectorize
def _sumw(y, j_l, j_u, logWmax, p, phi):
    j = np.arange(j_l, j_u + 1)
    sumw = np.sum(np.exp(_logWj(y, j, p, phi) - logWmax))
    return sumw