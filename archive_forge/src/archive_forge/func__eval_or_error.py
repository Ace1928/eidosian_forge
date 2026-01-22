import inspect
import io
import os
import platform
import warnings
import numpy
import cupy
import cupy_backends
def _eval_or_error(func, errors):
    try:
        return func()
    except errors as e:
        return repr(e)