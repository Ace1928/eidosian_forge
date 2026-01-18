import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_py3_decoded_valid_bytes(self):
    s = bytes('test-py2', 'utf-8')
    decoded_str = helpers.safe_decode_utf8(s)
    self.assertIsInstance(decoded_str, str)
    self.assertEqual(s, decoded_str.encode('utf-8'))