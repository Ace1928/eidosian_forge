import io
import time
import unittest
from fastimport import (
from :2
def test_unquote(self):
    s = b'hello \\"sweet\\" wo\\\\r\\tld'
    self.assertEqual(b'hello "sweet" wo\\r' + b'\tld', parser._unquote_c_string(s))