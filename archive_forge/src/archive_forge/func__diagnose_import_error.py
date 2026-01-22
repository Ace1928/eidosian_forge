import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _diagnose_import_error() -> str:
    msg = 'Failed to import CuPy.\n\nIf you installed CuPy via wheels (cupy-cudaXXX or cupy-rocm-X-X), make sure that the package matches with the version of CUDA or ROCm installed.\n\nOn Linux, you may need to set LD_LIBRARY_PATH environment variable depending on how you installed CUDA/ROCm.\nOn Windows, try setting CUDA_PATH environment variable.\n\nCheck the Installation Guide for details:\n  https://docs.cupy.dev/en/latest/install.html'
    if sys.platform == 'win32':
        try:
            msg += _diagnose_win32_dll_load()
        except Exception as e:
            msg += f'\n\nThe cause could not be identified: {type(e).__name__}: {e}'
    return msg