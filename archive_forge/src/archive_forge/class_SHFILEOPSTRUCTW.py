from __future__ import unicode_literals
import os.path as op
from send2trash.compat import text_type
from send2trash.util import preprocess_paths
from ctypes import (
from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL
class SHFILEOPSTRUCTW(Structure):
    _fields_ = [('hwnd', HWND), ('wFunc', UINT), ('pFrom', LPCWSTR), ('pTo', LPCWSTR), ('fFlags', c_uint), ('fAnyOperationsAborted', BOOL), ('hNameMappings', c_uint), ('lpszProgressTitle', LPCWSTR)]