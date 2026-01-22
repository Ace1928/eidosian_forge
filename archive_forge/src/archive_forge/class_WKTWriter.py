import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
class WKTWriter(IOBase):
    _constructor = wkt_writer_create
    ptr_type = WKT_WRITE_PTR
    destructor = wkt_writer_destroy
    _precision = None

    def __init__(self, dim=2, trim=False, precision=None):
        super().__init__()
        self._trim = DEFAULT_TRIM_VALUE
        self.trim = trim
        if precision is not None:
            self.precision = precision
        self.outdim = dim

    def write(self, geom):
        """Return the WKT representation of the given geometry."""
        return wkt_writer_write(self.ptr, geom.ptr)

    @property
    def outdim(self):
        return wkt_writer_get_outdim(self.ptr)

    @outdim.setter
    def outdim(self, new_dim):
        if new_dim not in (2, 3):
            raise ValueError('WKT output dimension must be 2 or 3')
        wkt_writer_set_outdim(self.ptr, new_dim)

    @property
    def trim(self):
        return self._trim

    @trim.setter
    def trim(self, flag):
        if bool(flag) != self._trim:
            self._trim = bool(flag)
            wkt_writer_set_trim(self.ptr, self._trim)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        if (not isinstance(precision, int) or precision < 0) and precision is not None:
            raise AttributeError('WKT output rounding precision must be non-negative integer or None.')
        if precision != self._precision:
            self._precision = precision
            wkt_writer_set_precision(self.ptr, -1 if precision is None else precision)