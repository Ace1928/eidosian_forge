from ctypes import byref, c_int
from datetime import date, datetime, time
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as capi
from django.utils.encoding import force_str
def as_double(self):
    """Retrieve the Field's value as a double (float)."""
    return capi.get_field_as_double(self._feat.ptr, self._index) if self.is_set else None