from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.geometry import GEOSGeometry, LinearGeometryMixin
from django.contrib.gis.geos.libgeos import GEOM_PTR
from django.contrib.gis.geos.linestring import LinearRing, LineString
from django.contrib.gis.geos.point import Point
from django.contrib.gis.geos.polygon import Polygon
def _get_single_internal(self, index):
    return capi.get_geomn(self.ptr, index)