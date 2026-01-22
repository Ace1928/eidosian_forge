import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
def _numpyarray_to_r(a, func):
    vec = func(numpy.ravel(a, order='F'))
    dim = ro.vectors.IntVector(a.shape)
    res = rinterface.baseenv['array'](vec, dim=dim)
    return res