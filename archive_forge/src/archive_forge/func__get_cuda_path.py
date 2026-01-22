import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _get_cuda_path():
    cuda_path = os.environ.get('CUDA_PATH', '')
    if os.path.exists(cuda_path):
        return cuda_path
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))
    if os.path.exists('/usr/local/cuda'):
        return '/usr/local/cuda'
    return None