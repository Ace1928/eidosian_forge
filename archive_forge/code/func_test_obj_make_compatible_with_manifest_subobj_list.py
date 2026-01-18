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
def test_obj_make_compatible_with_manifest_subobj_list(self):
    subobj = MyOwnedObject(baz=1)
    obj = MyObj(rel_objects=[subobj])
    obj.obj_relationships = {}
    manifest = {'MyOwnedObject': '1.2'}
    primitive = obj.obj_to_primitive()['versioned_object.data']
    method = 'obj_make_compatible_from_manifest'
    with mock.patch.object(subobj, method) as mock_compat:
        obj.obj_make_compatible_from_manifest(primitive, '1.5', manifest)
        mock_compat.assert_called_once_with(primitive['rel_objects'][0]['versioned_object.data'], '1.2', version_manifest=manifest)