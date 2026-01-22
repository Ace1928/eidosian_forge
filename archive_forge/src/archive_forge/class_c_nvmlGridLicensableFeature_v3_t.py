from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGridLicensableFeature_v3_t(_PrintableStructure):
    _fields_ = [('featureCode', _nvmlGridLicenseFeatureCode_t), ('featureState', c_uint), ('licenseInfo', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('productName', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('featureEnabled', c_uint)]