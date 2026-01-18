import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_object_compatibility(self):
    to_prim = mock.MagicMock(spec=callable)
    MyObject.VERSION = '1.1'
    MyObject.obj_to_primitive = to_prim
    self.ovc._test_object_compatibility(MyObject)
    expected_calls = [((), {'target_version': '1.0'}), ((), {'target_version': '1.1'})]
    self.assertEqual(expected_calls, to_prim.call_args_list, '_test_object_compatibility() did not test obj_to_primitive() on the correct target versions')