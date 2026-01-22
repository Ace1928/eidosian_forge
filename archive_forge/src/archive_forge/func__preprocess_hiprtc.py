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
def _preprocess_hiprtc(source, options):
    if _cuda_hip_version >= 40400000:
        code = '\n        // hiprtc segfaults if the input code is empty\n        __global__ void _cupy_preprocess_dummy_kernel_() { }\n        '
    else:
        code = '\n        // hiprtc segfaults if the input code is empty\n        #include <hip/hip_runtime.h>\n        __global__ void _cupy_preprocess_dummy_kernel_() { }\n        '
    prog = _NVRTCProgram(code)
    try:
        result, _ = prog.compile(options)
    except CompileException as e:
        dump = _get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            e.dump(sys.stderr)
        raise
    assert isinstance(result, bytes)
    return result