import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _load_freeimage(self):
    lib_names = ['freeimage', 'libfreeimage']
    exact_lib_names = ['FreeImage', 'libfreeimage.dylib', 'libfreeimage.so', 'libfreeimage.so.3']
    res_dirs = resource_dirs()
    plat = get_platform()
    if plat:
        fname = FNAME_PER_PLATFORM[plat]
        for dir in res_dirs:
            exact_lib_names.insert(0, os.path.join(dir, 'freeimage', fname))
    lib = os.getenv('IMAGEIO_FREEIMAGE_LIB', None)
    if lib is not None:
        exact_lib_names.insert(0, lib)
    try:
        lib, fname = load_lib(exact_lib_names, lib_names, res_dirs)
    except OSError as err:
        err_msg = str(err) + '\nPlease install the FreeImage library.'
        raise OSError(err_msg)
    self._lib = lib
    self.lib_fname = fname