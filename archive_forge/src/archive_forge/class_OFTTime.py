from ctypes import byref, c_int
from datetime import date, datetime, time
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as capi
from django.utils.encoding import force_str
class OFTTime(Field):

    @property
    def value(self):
        """Return a Python `time` object for this OFTTime field."""
        try:
            yy, mm, dd, hh, mn, ss, tz = self.as_datetime()
            return time(hh.value, mn.value, ss.value)
        except (ValueError, GDALException):
            return None