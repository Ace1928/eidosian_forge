import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
def _bytes_to_cf_data_ref(value: bytes) -> CFDataRef:
    return CoreFoundation.CFDataCreate(CoreFoundation.kCFAllocatorDefault, value, len(value))