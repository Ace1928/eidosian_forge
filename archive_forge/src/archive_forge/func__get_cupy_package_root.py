import inspect
import io
import os
import platform
import warnings
import numpy
import cupy
import cupy_backends
def _get_cupy_package_root(self):
    try:
        cupy_path = inspect.getfile(cupy)
    except TypeError:
        return None
    return os.path.dirname(cupy_path)