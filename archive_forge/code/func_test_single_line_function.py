import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
def test_single_line_function(self):

    @njit
    def foo():
        pass
    first_def_line = get_func_body_first_lineno(foo)
    self.assert_line_location(first_def_line, 2)