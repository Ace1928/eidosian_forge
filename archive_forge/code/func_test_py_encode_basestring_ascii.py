from unittest import TestCase
import simplejson.encoder
from simplejson.compat import b
def test_py_encode_basestring_ascii(self):
    self._test_encode_basestring_ascii(simplejson.encoder.py_encode_basestring_ascii)