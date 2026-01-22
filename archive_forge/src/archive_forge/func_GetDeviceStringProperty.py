from six.moves import queue
from six.moves import range
import ctypes
import ctypes.util
import logging
import sys
import threading
from pyu2f import errors
from pyu2f.hid import base
def GetDeviceStringProperty(dev_ref, key):
    """Reads string property from the HID device."""
    cf_key = CFStr(key)
    type_ref = iokit.IOHIDDeviceGetProperty(dev_ref, cf_key)
    cf.CFRelease(cf_key)
    if not type_ref:
        return None
    if cf.CFGetTypeID(type_ref) != cf.CFStringGetTypeID():
        raise errors.OsHidError('Expected string type, got {}'.format(cf.CFGetTypeID(type_ref)))
    type_ref = ctypes.cast(type_ref, CF_STRING_REF)
    out = ctypes.create_string_buffer(DEVICE_STRING_PROPERTY_BUFFER_SIZE)
    ret = cf.CFStringGetCString(type_ref, out, DEVICE_STRING_PROPERTY_BUFFER_SIZE, K_CF_STRING_ENCODING_UTF8)
    if not ret:
        return None
    return out.value.decode('utf8')