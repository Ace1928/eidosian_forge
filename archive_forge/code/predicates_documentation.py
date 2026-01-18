from ctypes import c_byte, c_char_p, c_double
from django.contrib.gis.geos.libgeos import GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import check_predicate
For GEOS binary predicate functions.