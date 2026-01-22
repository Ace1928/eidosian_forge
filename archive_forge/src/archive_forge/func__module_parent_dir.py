import io
import os
import pickle
import sys
import runpy
import types
import warnings
from . import get_start_method, set_start_method
from . import process
from . import util
def _module_parent_dir(mod):
    dir, filename = os.path.split(_module_dir(mod))
    if dir == os.curdir or not dir:
        dir = os.getcwd()
    return dir