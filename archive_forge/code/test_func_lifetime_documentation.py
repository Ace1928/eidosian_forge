import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest

        When a jitted function calls into another jitted function, check
        that everything is collected as desired.
        