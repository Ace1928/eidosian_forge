from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def _add_core(value, __setup, events, errors):
    if not issubclass(value, Extension):
        raise XcffibException('Extension type not derived from xcffib.Extension')
    if not issubclass(__setup, Struct):
        raise XcffibException('Setup type not derived from xcffib.Struct')
    global core
    global core_events
    global core_errors
    global _setup
    core = value
    core_events = events
    core_errors = errors
    _setup = __setup