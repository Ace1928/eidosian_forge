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
def test_object_version_mappings(self):
    self.skipTest('this needs to be generalized')
    for obj_classes in base.VersionedObjectRegistry.obj_classes().values():
        for obj_class in obj_classes:
            if issubclass(obj_class, base.ObjectListBase):
                self._test_object_list_version_mappings(obj_class)