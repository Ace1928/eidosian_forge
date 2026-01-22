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
@base.VersionedObjectRegistry.register
class TestSubclassedObject(RandomMixInWithNoFields, MyObj):
    fields = {'new_field': fields.Field(fields.String())}
    child_versions = {'1.0': '1.0', '1.1': '1.1', '1.2': '1.1', '1.3': '1.2', '1.4': '1.3', '1.5': '1.4', '1.6': '1.5', '1.7': '1.6'}