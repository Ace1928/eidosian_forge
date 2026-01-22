import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def _calc_mcc(self, cmat):
    n = cmat.sum()
    x = cmat.sum(axis=1)
    y = cmat.sum(axis=0)
    cov_xx = numpy.sum(x * (n - x))
    cov_yy = numpy.sum(y * (n - y))
    if cov_xx == 0 or cov_yy == 0:
        return float('nan')
    i = cmat.diagonal()
    cov_xy = numpy.sum(i * n - x * y)
    return cov_xy / (cov_xx * cov_yy) ** 0.5