import re
from ctypes import addressof, byref, c_double
from django.contrib.gis import gdal
from django.contrib.gis.geometry import hex_regex, json_regex, wkt_regex
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.coordseq import GEOSCoordSeq
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOM_PTR, geos_version_tuple
from django.contrib.gis.geos.mutable_list import ListMixin
from django.contrib.gis.geos.prepared import PreparedGeometry
from django.contrib.gis.geos.prototypes.io import ewkb_w, wkb_r, wkb_w, wkt_r, wkt_w
from django.utils.deconstruct import deconstructible
from django.utils.encoding import force_bytes, force_str
class LinearGeometryMixin:
    """
    Used for LineString and MultiLineString.
    """

    def interpolate(self, distance):
        return self._topology(capi.geos_interpolate(self.ptr, distance))

    def interpolate_normalized(self, distance):
        return self._topology(capi.geos_interpolate_normalized(self.ptr, distance))

    def project(self, point):
        from .point import Point
        if not isinstance(point, Point):
            raise TypeError('locate_point argument must be a Point')
        return capi.geos_project(self.ptr, point.ptr)

    def project_normalized(self, point):
        from .point import Point
        if not isinstance(point, Point):
            raise TypeError('locate_point argument must be a Point')
        return capi.geos_project_normalized(self.ptr, point.ptr)

    @property
    def merged(self):
        """
        Return the line merge of this Geometry.
        """
        return self._topology(capi.geos_linemerge(self.ptr))

    @property
    def closed(self):
        """
        Return whether or not this Geometry is closed.
        """
        return capi.geos_isclosed(self.ptr)