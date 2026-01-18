from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def void_output(func, argtypes, errcheck=True, cpl=False):
    """
    For functions that don't only return an error code that needs to
    be examined.
    """
    if argtypes:
        func.argtypes = argtypes
    if errcheck:
        func.restype = c_int
        func.errcheck = partial(check_errcode, cpl=cpl)
    else:
        func.restype = None
    return func