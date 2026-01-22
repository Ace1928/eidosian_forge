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
class MKLController(LibController):
    """Controller class for MKL"""
    user_api = 'blas'
    internal_api = 'mkl'
    filename_prefixes = ('libmkl_rt', 'mkl_rt', 'libblas')
    check_symbols = ('MKL_Get_Max_Threads', 'MKL_Set_Num_Threads', 'MKL_Get_Version_String', 'MKL_Set_Threading_Layer')

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, 'MKL_Get_Max_Threads', lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, 'MKL_Set_Num_Threads', lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        if not hasattr(self.dynlib, 'MKL_Get_Version_String'):
            return None
        res = ctypes.create_string_buffer(200)
        self.dynlib.MKL_Get_Version_String(res, 200)
        version = res.value.decode('utf-8')
        group = re.search('Version ([^ ]+) ', version)
        if group is not None:
            version = group.groups()[0]
        return version.strip()

    def _get_threading_layer(self):
        """Return the threading layer of MKL"""
        set_threading_layer = getattr(self.dynlib, 'MKL_Set_Threading_Layer', lambda layer: -1)
        layer_map = {0: 'intel', 1: 'sequential', 2: 'pgi', 3: 'gnu', 4: 'tbb', -1: 'not specified'}
        return layer_map[set_threading_layer(-1)]