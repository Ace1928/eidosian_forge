import contextlib
import os
import sklearn
from .._min_dependencies import CYTHON_MIN_VERSION
from ..externals._packaging.version import parse
from .openmp_helpers import check_openmp_support
from .pre_build_helpers import basic_check_build
def _check_cython_version():
    message = 'Please install Cython with a version >= {0} in order to build a scikit-learn from source.'.format(CYTHON_MIN_VERSION)
    try:
        import Cython
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(message) from e
    if parse(Cython.__version__) < parse(CYTHON_MIN_VERSION):
        message += ' The current version of Cython is {} installed in {}.'.format(Cython.__version__, Cython.__path__)
        raise ValueError(message)