import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _valid_error_name(name):
    """Check whether name is a valid error name."""
    return all((x.isalnum() or x in '_.' for x in name))