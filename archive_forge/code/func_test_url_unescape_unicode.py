import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_url_unescape_unicode(self):
    tests = [('%C3%A9', 'é', 'utf8'), ('%C3%A9', 'Ã©', 'latin1'), ('%C3%A9', utf8('é'), None)]
    for escaped, unescaped, encoding in tests:
        self.assertEqual(url_unescape(to_unicode(escaped), encoding), unescaped)
        self.assertEqual(url_unescape(utf8(escaped), encoding), unescaped)