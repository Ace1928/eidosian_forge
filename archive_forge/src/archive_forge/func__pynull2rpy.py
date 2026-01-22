from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _pynull2rpy(pyobject):
    if pyobject is None:
        pyobject = rpy2.rinterface.NULL
    return pyobject