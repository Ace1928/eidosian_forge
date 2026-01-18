import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.core import config
from numba.tests.support import TestCase

    Tests to assert that code which should vectorize does indeed vectorize
    