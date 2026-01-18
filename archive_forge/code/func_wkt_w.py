import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
def wkt_w(dim=2, trim=False, precision=None):
    if not thread_context.wkt_w:
        thread_context.wkt_w = WKTWriter(dim=dim, trim=trim, precision=precision)
    else:
        thread_context.wkt_w.outdim = dim
        thread_context.wkt_w.trim = trim
        thread_context.wkt_w.precision = precision
    return thread_context.wkt_w