from ctypes import byref, c_int
from datetime import date, datetime, time
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as capi
from django.utils.encoding import force_str
class OFTInteger(Field):
    _bit64 = False

    @property
    def value(self):
        """Return an integer contained in this field."""
        return self.as_int(self._bit64)

    @property
    def type(self):
        """
        GDAL uses OFTReals to represent OFTIntegers in created
        shapefiles -- forcing the type here since the underlying field
        type may actually be OFTReal.
        """
        return 0