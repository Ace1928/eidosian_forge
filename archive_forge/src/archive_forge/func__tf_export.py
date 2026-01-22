import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _tf_export(*x, **kwargs):
    del x, kwargs
    return lambda x: x