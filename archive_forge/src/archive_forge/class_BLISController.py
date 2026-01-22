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
class BLISController(LibController):
    """Controller class for BLIS"""
    user_api = 'blas'
    internal_api = 'blis'
    filename_prefixes = ('libblis', 'libblas')
    check_symbols = ('bli_thread_get_num_threads', 'bli_thread_set_num_threads', 'bli_info_get_version_str', 'bli_info_get_enable_openmp', 'bli_info_get_enable_pthreads', 'bli_arch_query_id', 'bli_arch_string')

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, 'bli_thread_get_num_threads', lambda: None)
        num_threads = get_func()
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, 'bli_thread_set_num_threads', lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        get_version_ = getattr(self.dynlib, 'bli_info_get_version_str', None)
        if get_version_ is None:
            return None
        get_version_.restype = ctypes.c_char_p
        return get_version_().decode('utf-8')

    def _get_threading_layer(self):
        """Return the threading layer of BLIS"""
        if getattr(self.dynlib, 'bli_info_get_enable_openmp', lambda: False)():
            return 'openmp'
        elif getattr(self.dynlib, 'bli_info_get_enable_pthreads', lambda: False)():
            return 'pthreads'
        return 'disabled'

    def _get_architecture(self):
        """Return the architecture detected by BLIS"""
        bli_arch_query_id = getattr(self.dynlib, 'bli_arch_query_id', None)
        bli_arch_string = getattr(self.dynlib, 'bli_arch_string', None)
        if bli_arch_query_id is None or bli_arch_string is None:
            return None
        bli_arch_query_id.restype = ctypes.c_int
        bli_arch_string.restype = ctypes.c_char_p
        return bli_arch_string(bli_arch_query_id()).decode('utf-8')