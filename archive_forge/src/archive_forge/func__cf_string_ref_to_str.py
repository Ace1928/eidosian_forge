import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
def _cf_string_ref_to_str(cf_string_ref: CFStringRef) -> str | None:
    """
    Creates a Unicode string from a CFString object. Used entirely for error
    reporting.
    Yes, it annoys me quite a lot that this function is this complex.
    """
    string = CoreFoundation.CFStringGetCStringPtr(cf_string_ref, CFConst.kCFStringEncodingUTF8)
    if string is None:
        buffer = ctypes.create_string_buffer(1024)
        result = CoreFoundation.CFStringGetCString(cf_string_ref, buffer, 1024, CFConst.kCFStringEncodingUTF8)
        if not result:
            raise OSError('Error copying C string from CFStringRef')
        string = buffer.value
    if string is not None:
        string = string.decode('utf-8')
    return string