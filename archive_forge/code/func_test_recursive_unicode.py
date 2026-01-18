import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_recursive_unicode(self):
    tests = {'dict': {b'foo': b'bar'}, 'list': [b'foo', b'bar'], 'tuple': (b'foo', b'bar'), 'bytes': b'foo'}
    self.assertEqual(recursive_unicode(tests['dict']), {'foo': 'bar'})
    self.assertEqual(recursive_unicode(tests['list']), ['foo', 'bar'])
    self.assertEqual(recursive_unicode(tests['tuple']), ('foo', 'bar'))
    self.assertEqual(recursive_unicode(tests['bytes']), 'foo')