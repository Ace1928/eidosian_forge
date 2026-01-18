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
def test_obj_make_compatible_complains_about_missing_rel_rules(self):
    subobj = MyOwnedObject(baz=1)
    obj = MyObj(foo=123, rel_object=subobj)
    obj.obj_relationships = {}
    self.assertRaises(exception.ObjectActionError, obj.obj_make_compatible, {}, '1.0')