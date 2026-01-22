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
def _test_object_list_version_mappings(self, list_obj_class):
    list_field = list_obj_class.fields['objects']
    item_obj_field = list_field._type._element_type
    item_obj_name = item_obj_field._type._obj_name
    obj_classes = base.VersionedObjectRegistry.obj_classes()[item_obj_name]
    for item_class in obj_classes:
        if is_test_object(item_class):
            continue
        self.assertIn(item_class.VERSION, list_obj_class.child_versions.values(), 'Version mapping is incomplete for %s' % list_obj_class.__name__)