from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager
import sys  # Note: the sys import must be here anyways (others depend on it)
import codecs as _codecs
import os
from _pydevd_bundle import pydevd_vm_type
from _pydev_bundle._pydev_saved_modules import thread, threading
def as_float_in_env(env_key, default):
    value = os.getenv(env_key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        raise RuntimeError('Error: expected the env variable: %s to be set to a float value. Found: %s' % (env_key, value))