from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def _check_str(result, func, cargs):
    res = check_string(result, func, cargs, offset=offset, str_result=str_result)
    if res and decoding:
        res = res.decode(decoding)
    return res