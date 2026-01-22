from ctypes import c_uint
from django.contrib.gis import gdal
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry
def _set_list(self, length, items):
    ptr = self._create_point(length, items)
    if ptr:
        srid = self.srid
        capi.destroy_geom(self.ptr)
        self._ptr = ptr
        if srid is not None:
            self.srid = srid
        self._post_init()
    else:
        raise GEOSException('Geometry resulting from slice deletion was invalid.')