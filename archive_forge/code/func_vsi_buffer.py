import json
import os
import sys
import uuid
from ctypes import (
from pathlib import Path
from django.contrib.gis.gdal.driver import Driver
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.band import BandList
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.gdal.raster.const import (
from django.contrib.gis.gdal.srs import SpatialReference, SRSException
from django.contrib.gis.geometry import json_regex
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
@property
def vsi_buffer(self):
    if not (self.is_vsi_based and self.name.startswith(VSI_MEM_FILESYSTEM_BASE_PATH)):
        return None
    out_length = c_int()
    dat = capi.get_mem_buffer_from_vsi_file(force_bytes(self.name), byref(out_length), VSI_DELETE_BUFFER_ON_READ)
    return string_at(dat, out_length.value)