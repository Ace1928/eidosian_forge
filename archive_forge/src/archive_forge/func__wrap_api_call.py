import ctypes
import functools
import sys
from numba.core import config
from numba.cuda.cudadrv.driver import ERROR_MAP, make_logger
from numba.cuda.cudadrv.error import CudaSupportError, CudaRuntimeError
from numba.cuda.cudadrv.libs import open_cudalib
from numba.cuda.cudadrv.rtapi import API_PROTOTYPES
from numba.cuda.cudadrv import enums
def _wrap_api_call(self, fname, libfn):

    @functools.wraps(libfn)
    def safe_cuda_api_call(*args):
        _logger.debug('call runtime api: %s', libfn.__name__)
        retcode = libfn(*args)
        self._check_error(fname, retcode)
    return safe_cuda_api_call