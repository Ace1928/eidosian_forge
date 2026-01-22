import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _check_cusolver_dev_info_if_synchronization_allowed(routine, dev_info):
    assert isinstance(dev_info, _core.ndarray)
    config_linalg = cupyx._ufunc_config.get_config_linalg()
    if config_linalg == 'ignore':
        return
    try:
        name = routine.__name__
    except AttributeError:
        name = routine
    assert config_linalg == 'raise'
    if (dev_info != 0).any():
        raise linalg.LinAlgError('Error reported by {} in cuSOLVER. devInfo = {}. Please refer to the cuSOLVER documentation.'.format(name, dev_info))