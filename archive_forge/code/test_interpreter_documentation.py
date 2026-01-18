import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (

            Dictionary update between two constant
            dictionaries. This verifies d2 doesn't
            get incorrectly removed.
            