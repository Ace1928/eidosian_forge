from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def voidptr_output(func, argtypes, errcheck=True):
    """For functions that return c_void_p."""
    func.argtypes = argtypes
    func.restype = c_void_p
    if errcheck:
        func.errcheck = check_pointer
    return func