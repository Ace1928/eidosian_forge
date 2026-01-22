from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _pysce2rpy(pyobject):
    if utils._try_import('anndata2ri'):
        pyobject = anndata2ri.py2rpy(pyobject)
    return pyobject