from ctypes import c_uint
from django.contrib.gis import gdal
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry
def _from_pickle_wkb(self, wkb):
    return self._create_empty() if wkb is None else super()._from_pickle_wkb(wkb)