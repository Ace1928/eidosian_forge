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
class OpenBLASController(LibController):
    """Controller class for OpenBLAS"""
    user_api = 'blas'
    internal_api = 'openblas'
    filename_prefixes = ('libopenblas', 'libblas')
    check_symbols = ('openblas_get_num_threads', 'openblas_get_num_threads64_', 'openblas_set_num_threads', 'openblas_set_num_threads64_', 'openblas_get_config', 'openblas_get_config64_', 'openblas_get_parallel', 'openblas_get_parallel64_', 'openblas_get_corename', 'openblas_get_corename64_')

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, 'openblas_get_num_threads', getattr(self.dynlib, 'openblas_get_num_threads64_', lambda: None))
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, 'openblas_set_num_threads', getattr(self.dynlib, 'openblas_set_num_threads64_', lambda num_threads: None))
        return set_func(num_threads)

    def get_version(self):
        get_config = getattr(self.dynlib, 'openblas_get_config', getattr(self.dynlib, 'openblas_get_config64_', None))
        if get_config is None:
            return None
        get_config.restype = ctypes.c_char_p
        config = get_config().split()
        if config[0] == b'OpenBLAS':
            return config[1].decode('utf-8')
        return None

    def _get_threading_layer(self):
        """Return the threading layer of OpenBLAS"""
        openblas_get_parallel = getattr(self.dynlib, 'openblas_get_parallel', getattr(self.dynlib, 'openblas_get_parallel64_', None))
        if openblas_get_parallel is None:
            return 'unknown'
        threading_layer = openblas_get_parallel()
        if threading_layer == 2:
            return 'openmp'
        elif threading_layer == 1:
            return 'pthreads'
        return 'disabled'

    def _get_architecture(self):
        """Return the architecture detected by OpenBLAS"""
        get_corename = getattr(self.dynlib, 'openblas_get_corename', getattr(self.dynlib, 'openblas_get_corename64_', None))
        if get_corename is None:
            return None
        get_corename.restype = ctypes.c_char_p
        return get_corename().decode('utf-8')