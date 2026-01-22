from ctypes import byref, c_int
from datetime import date, datetime, time
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as capi
from django.utils.encoding import force_str
def as_datetime(self):
    """Retrieve the Field's value as a tuple of date & time components."""
    if not self.is_set:
        return None
    yy, mm, dd, hh, mn, ss, tz = [c_int() for i in range(7)]
    status = capi.get_field_as_datetime(self._feat.ptr, self._index, byref(yy), byref(mm), byref(dd), byref(hh), byref(mn), byref(ss), byref(tz))
    if status:
        return (yy, mm, dd, hh, mn, ss, tz)
    else:
        raise GDALException('Unable to retrieve date & time information from the field.')