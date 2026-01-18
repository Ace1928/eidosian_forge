import time
import ctypes
import numpy as np
from numba.tests.support import captured_stdout
from numba import vectorize, guvectorize
import unittest

        Testing similar issue to #1998 due to GIL reacquiring for Gufunc
        