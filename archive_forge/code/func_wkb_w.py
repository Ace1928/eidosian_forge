import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
def wkb_w(dim=2):
    if not thread_context.wkb_w:
        thread_context.wkb_w = WKBWriter(dim=dim)
    else:
        thread_context.wkb_w.outdim = dim
    return thread_context.wkb_w