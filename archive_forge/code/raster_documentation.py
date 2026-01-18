from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_void_p
from functools import partial
from django.contrib.gis.gdal.libgdal import std_call
from django.contrib.gis.gdal.prototypes.generation import (

This module houses the ctypes function prototypes for GDAL DataSource (raster)
related data structures.
