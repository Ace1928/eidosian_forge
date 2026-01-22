import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
class WKBWriter(IOBase):
    _constructor = wkb_writer_create
    ptr_type = WKB_WRITE_PTR
    destructor = wkb_writer_destroy
    geos_version = geos_version_tuple()

    def __init__(self, dim=2):
        super().__init__()
        self.outdim = dim

    def _handle_empty_point(self, geom):
        from django.contrib.gis.geos import Point
        if isinstance(geom, Point) and geom.empty:
            if self.srid:
                geom = Point(float('NaN'), float('NaN'), srid=geom.srid)
            else:
                raise ValueError('Empty point is not representable in WKB.')
        return geom

    def write(self, geom):
        """Return the WKB representation of the given geometry."""
        geom = self._handle_empty_point(geom)
        wkb = wkb_writer_write(self.ptr, geom.ptr, byref(c_size_t()))
        return memoryview(wkb)

    def write_hex(self, geom):
        """Return the HEXEWKB representation of the given geometry."""
        geom = self._handle_empty_point(geom)
        wkb = wkb_writer_write_hex(self.ptr, geom.ptr, byref(c_size_t()))
        return wkb

    def _get_byteorder(self):
        return wkb_writer_get_byteorder(self.ptr)

    def _set_byteorder(self, order):
        if order not in (0, 1):
            raise ValueError('Byte order parameter must be 0 (Big Endian) or 1 (Little Endian).')
        wkb_writer_set_byteorder(self.ptr, order)
    byteorder = property(_get_byteorder, _set_byteorder)

    @property
    def outdim(self):
        return wkb_writer_get_outdim(self.ptr)

    @outdim.setter
    def outdim(self, new_dim):
        if new_dim not in (2, 3):
            raise ValueError('WKB output dimension must be 2 or 3')
        wkb_writer_set_outdim(self.ptr, new_dim)

    @property
    def srid(self):
        return bool(wkb_writer_get_include_srid(self.ptr))

    @srid.setter
    def srid(self, include):
        wkb_writer_set_include_srid(self.ptr, bool(include))