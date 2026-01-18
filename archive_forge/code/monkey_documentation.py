import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist

        Prepare the parameters for patch_func to patch indicated function.
        