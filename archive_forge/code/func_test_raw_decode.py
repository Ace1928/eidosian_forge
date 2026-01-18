from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_raw_decode(self):
    cls = json.decoder.JSONDecoder
    self.assertEqual(({'a': {}}, 9), cls().raw_decode('{"a": {}}'))
    self.assertEqual(({'a': {}}, 9), cls(object_pairs_hook=dict).raw_decode('{"a": {}}'))
    self.assertEqual(({'a': {}}, 11), cls().raw_decode(' \n{"a": {}}'))