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
def test_test_object_compatibility_args_kwargs(self):
    to_prim = mock.MagicMock(spec=callable)
    MyObject.obj_to_primitive = to_prim
    MyObject.VERSION = '1.1'
    args = [1]
    kwargs = {'foo': 'bar'}
    with mock.patch.object(MyObject, '__init__', return_value=None) as mock_init:
        self.ovc._test_object_compatibility(MyObject, init_args=args, init_kwargs=kwargs)
    expected_init = ((1,), {'foo': 'bar'})
    expected_init_calls = [expected_init, expected_init]
    self.assertEqual(expected_init_calls, mock_init.call_args_list, '_test_object_compatibility() did not call __init__() properly on the object')
    expected_to_prim = [((), {'target_version': '1.0'}), ((), {'target_version': '1.1'})]
    self.assertEqual(expected_to_prim, to_prim.call_args_list, '_test_object_compatibility() did not test obj_to_primitive() on the correct target versions')