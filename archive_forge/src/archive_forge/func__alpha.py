import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def _alpha(p):
    return (2 - p) / (1 - p)