import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
@_util.memoize()
def _get_max_compute_capability():
    major, minor = _get_nvrtc_version()
    if major < 11:
        nvrtc_max_compute_capability = '75'
    elif major == 11 and minor == 0:
        nvrtc_max_compute_capability = '80'
    elif major == 11 and minor < 8:
        nvrtc_max_compute_capability = '86'
    else:
        nvrtc_max_compute_capability = '90'
    return nvrtc_max_compute_capability