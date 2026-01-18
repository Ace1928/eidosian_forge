from ctypes import POINTER, c_char_p, c_int, c_void_p
from django.contrib.gis.gdal.libgdal import GDAL_VERSION, lgdal, std_call
from django.contrib.gis.gdal.prototypes.generation import (
from_wkt = void_output(lgdal.OSRImportFromWkt, [c_void_p, POINTER(c_char_p)])
from_proj = void_output(lgdal.OSRImportFromProj4, [c_void_p, c_char_p])
from_epsg = void_output(std_call("OSRImportFromEPSG"), [c_void_p, c_int])
from_xml = void_output(lgdal.OSRImportFromXML, [c_void_p, c_char_p])
from_user_input = void_output(std_call("OSRSetFromUserInput"), [c_void_p, c_char_p])

    Create a ctypes function prototype for OSR units functions, e.g.,
    OSRGetAngularUnits, OSRGetLinearUnits.
    