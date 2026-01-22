import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def _find_libraries_with_dyld(self):
    """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on OSX system only
        """
    libc = self._get_libc()
    if not hasattr(libc, '_dyld_image_count'):
        return []
    n_dyld = libc._dyld_image_count()
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    for i in range(n_dyld):
        filepath = ctypes.string_at(libc._dyld_get_image_name(i))
        filepath = filepath.decode('utf-8')
        self._make_controller_from_path(filepath)