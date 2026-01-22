import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
def _system():
    system = platform.system().lower()
    for c in '.-_':
        system = system.split(c)[0]
    return system