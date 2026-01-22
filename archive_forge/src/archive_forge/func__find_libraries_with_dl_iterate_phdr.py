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
def _find_libraries_with_dl_iterate_phdr(self):
    """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developer @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """
    libc = self._get_libc()
    if not hasattr(libc, 'dl_iterate_phdr'):
        return []

    def match_library_callback(info, size, data):
        filepath = info.contents.dlpi_name
        if filepath:
            filepath = filepath.decode('utf-8')
            self._make_controller_from_path(filepath)
        return 0
    c_func_signature = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_dl_phdr_info), ctypes.c_size_t, ctypes.c_char_p)
    c_match_library_callback = c_func_signature(match_library_callback)
    data = ctypes.c_char_p(b'')
    libc.dl_iterate_phdr(c_match_library_callback, data)