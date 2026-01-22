import contextlib
import ctypes
import os
import platform
import subprocess
import sys
import time
import warnings
from ctypes import c_size_t, sizeof, c_wchar_p, get_errno, c_wchar
def _executable_exists(name):
    return subprocess.call([WHICH_CMD, name], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0