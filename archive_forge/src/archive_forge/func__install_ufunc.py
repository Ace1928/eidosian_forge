import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
def _install_ufunc(func_name):

    def f(self):
        ufunc = getattr(cupy, func_name)
        result = ufunc(self.data)
        return self._with_data(result)
    f.__doc__ = 'Elementwise %s.' % func_name
    f.__name__ = func_name
    setattr(_data_matrix, func_name, f)