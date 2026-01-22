from ctypes import c_uint
from django.contrib.gis import gdal
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry
def _get_single_external(self, index):
    if index == 0:
        return self.x
    elif index == 1:
        return self.y
    elif index == 2:
        return self.z