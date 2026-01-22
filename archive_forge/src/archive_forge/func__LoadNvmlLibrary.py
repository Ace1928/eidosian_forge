from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def _LoadNvmlLibrary():
    """
    Load the library if it isn't loaded already
    """
    global nvmlLib
    if nvmlLib == None:
        libLoadLock.acquire()
        try:
            if nvmlLib == None:
                try:
                    if sys.platform[:3] == 'win':
                        try:
                            nvmlLib = CDLL(os.path.join(os.getenv('WINDIR', 'C:/Windows'), 'System32/nvml.dll'))
                        except OSError as ose:
                            nvmlLib = CDLL(os.path.join(os.getenv('ProgramFiles', 'C:/Program Files'), 'NVIDIA Corporation/NVSMI/nvml.dll'))
                    else:
                        nvmlLib = CDLL('libnvidia-ml.so.1')
                except OSError as ose:
                    _nvmlCheckReturn(NVML_ERROR_LIBRARY_NOT_FOUND)
                if nvmlLib == None:
                    _nvmlCheckReturn(NVML_ERROR_LIBRARY_NOT_FOUND)
        finally:
            libLoadLock.release()