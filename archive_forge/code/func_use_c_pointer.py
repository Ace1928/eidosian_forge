from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def use_c_pointer(x):
    """
    Running in Python will cause a segfault.
    """
    threadstate = savethread()
    x += 1
    restorethread(threadstate)
    return x