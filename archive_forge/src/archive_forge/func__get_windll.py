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
@classmethod
def _get_windll(cls, dll_name):
    """Load a windows DLL"""
    dll = cls._system_libraries.get(dll_name)
    if dll is None:
        dll = ctypes.WinDLL(f'{dll_name}.dll')
        cls._system_libraries[dll_name] = dll
    return dll