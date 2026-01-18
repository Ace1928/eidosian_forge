from io import StringIO
import logging
import unittest
from numba.core import tracing
def test_static_method(self):
    with self.capture:
        Class.static_method()
    self.assertEqual(self.capture.getvalue(), '>> static_method()\n' + '<< static_method\n')