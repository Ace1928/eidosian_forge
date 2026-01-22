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
class OpenMPController(LibController):
    """Controller class for OpenMP"""
    user_api = 'openmp'
    internal_api = 'openmp'
    filename_prefixes = ('libiomp', 'libgomp', 'libomp', 'vcomp')
    check_symbols = ('omp_get_max_threads', 'omp_get_num_threads')

    def get_num_threads(self):
        get_func = getattr(self.dynlib, 'omp_get_max_threads', lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, 'omp_set_num_threads', lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        return None