import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
def _load_framework(name):
    return ctypes.cdll.LoadLibrary(ctypes.util.find_library(name))