from ctypes import byref, c_int
from datetime import date, datetime, time
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as capi
from django.utils.encoding import force_str
class OFTWideStringList(Field):
    pass