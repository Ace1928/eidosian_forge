import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _check_cublas_info_array_if_synchronization_allowed(routine, info_array):
    assert isinstance(info_array, _core.ndarray)
    assert info_array.ndim == 1
    config_linalg = cupyx._ufunc_config.get_config_linalg()
    if config_linalg == 'ignore':
        return
    assert config_linalg == 'raise'
    if (info_array != 0).any():
        raise linalg.LinAlgError('Error reported by {} in cuBLAS. infoArray/devInfoArray = {}. Please refer to the cuBLAS documentation.'.format(routine.__name__, info_array))