from ctypes import POINTER, c_char_p, c_double, c_int, c_void_p
from django.contrib.gis.gdal.envelope import OGREnvelope
from django.contrib.gis.gdal.libgdal import lgdal
from django.contrib.gis.gdal.prototypes.errcheck import check_envelope
from django.contrib.gis.gdal.prototypes.generation import (
from_json = geom_output(lgdal.OGR_G_CreateGeometryFromJson, [c_char_p])
from_wkb = geom_output(
from_wkt = geom_output(
from_gml = geom_output(lgdal.OGR_G_CreateFromGML, [c_char_p])
import_wkt = void_output(lgdal.OGR_G_ImportFromWkt, [c_void_p, POINTER(c_char_p)])
def topology_func(f):
    f.argtypes = [c_void_p, c_void_p]
    f.restype = c_int
    f.errcheck = lambda result, func, cargs: bool(result)
    return f