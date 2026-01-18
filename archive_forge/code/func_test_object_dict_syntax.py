import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_object_dict_syntax(self):
    obj = MyObj(foo=123, bar='text')
    self.assertEqual(obj['foo'], 123)
    self.assertIn('bar', obj)
    self.assertNotIn('missing', obj)
    self.assertEqual(sorted(iter(obj)), ['bar', 'foo'])
    self.assertEqual(sorted(obj.keys()), ['bar', 'foo'])
    self.assertEqual(sorted(obj.values(), key=str), [123, 'text'])
    self.assertEqual(sorted(obj.items()), [('bar', 'text'), ('foo', 123)])
    self.assertEqual(dict(obj), {'foo': 123, 'bar': 'text'})