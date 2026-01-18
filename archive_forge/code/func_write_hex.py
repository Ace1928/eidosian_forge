import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
def write_hex(self, geom):
    """Return the HEXEWKB representation of the given geometry."""
    geom = self._handle_empty_point(geom)
    wkb = wkb_writer_write_hex(self.ptr, geom.ptr, byref(c_size_t()))
    return wkb