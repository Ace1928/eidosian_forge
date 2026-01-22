from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _rpylist2py(robject):
    if not isinstance(robject, rpy2.robjects.vectors.ListVector):
        raise NotImplementedError
    names = rpy2py(robject.names)
    if names is None or len(names) > len(np.unique(names)):
        robject = np.array([rpy2py(obj) for obj in robject])
    else:
        robject = {name: rpy2py(obj) for name, obj in zip(robject.names, robject)}
    return robject