import sys
from binascii import b2a_hex
from ctypes import byref, c_char_p, c_double, c_ubyte, c_void_p, string_at
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.envelope import Envelope, OGREnvelope
from django.contrib.gis.gdal.error import GDALException, SRSException
from django.contrib.gis.gdal.geomtype import OGRGeomType
from django.contrib.gis.gdal.prototypes import geom as capi
from django.contrib.gis.gdal.prototypes import srs as srs_api
from django.contrib.gis.gdal.srs import CoordTransform, SpatialReference
from django.contrib.gis.geometry import hex_regex, json_regex, wkt_regex
from django.utils.encoding import force_bytes
def _set_srs(self, srs):
    """Set the SpatialReference for this geometry."""
    if isinstance(srs, SpatialReference):
        srs_ptr = srs.ptr
    elif isinstance(srs, (int, str)):
        sr = SpatialReference(srs)
        srs_ptr = sr.ptr
    elif srs is None:
        srs_ptr = None
    else:
        raise TypeError('Cannot assign spatial reference with object of type: %s' % type(srs))
    capi.assign_srs(self.ptr, srs_ptr)