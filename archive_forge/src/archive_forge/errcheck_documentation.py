from ctypes import c_void_p, string_at
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOSFuncFactory

    Error checking for routines that return strings.

    This frees the memory allocated by GEOS at the result pointer.
    