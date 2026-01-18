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
def test_obj_make_compatible_hits_sub_objects_with_rels(self):
    subobj = MyOwnedObject(baz=1)
    obj = MyObj(foo=123, rel_object=subobj)
    obj.obj_relationships = {'rel_object': [('1.0', '1.0')]}
    with mock.patch.object(obj, '_obj_make_obj_compatible') as mock_compat:
        obj.obj_make_compatible({'rel_object': 'foo'}, '1.10')
        mock_compat.assert_called_once_with({'rel_object': 'foo'}, '1.10', 'rel_object')