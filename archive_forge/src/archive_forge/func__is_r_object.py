from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _is_r_object(obj):
    return 'rpy2.robjects' in str(type(obj)) or 'rpy2.rinterface' in str(type(obj))