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
def _set_threadpool_limits(self):
    """Change the maximal number of threads in selected thread pools.

        Return a list with all the supported libraries that have been found
        matching `self._prefixes` and `self._user_api`.
        """
    if self._limits is None:
        return
    for lib_controller in self._controller.lib_controllers:
        if lib_controller.prefix in self._limits:
            num_threads = self._limits[lib_controller.prefix]
        elif lib_controller.user_api in self._limits:
            num_threads = self._limits[lib_controller.user_api]
        else:
            continue
        if num_threads is not None:
            lib_controller.set_num_threads(num_threads)