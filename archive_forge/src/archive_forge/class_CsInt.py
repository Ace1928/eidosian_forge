from ctypes import POINTER, c_byte, c_double, c_int, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import GEOSException, last_arg_byref
class CsInt(GEOSFuncFactory):
    """For coordinate sequence routines that return an integer."""
    argtypes = [CS_PTR, POINTER(c_uint)]
    restype = c_int
    errcheck = staticmethod(check_cs_get)