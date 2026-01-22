import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, 'ThreadpoolController'):
        return None
    if not hasattr(sklearn, '_sklearn_threadpool_controller'):
        sklearn._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()
    return sklearn._sklearn_threadpool_controller