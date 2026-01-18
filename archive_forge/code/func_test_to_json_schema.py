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
def test_to_json_schema(self):
    schema = self.FakeObject.to_json_schema()
    self.assertEqual({'$schema': 'http://json-schema.org/draft-04/schema#', 'title': 'FakeObject', 'type': ['object'], 'properties': {'versioned_object.namespace': {'type': 'string'}, 'versioned_object.name': {'type': 'string'}, 'versioned_object.version': {'type': 'string'}, 'versioned_object.changes': {'type': 'array', 'items': {'type': 'string'}}, 'versioned_object.data': {'type': 'object', 'description': 'fields of FakeObject', 'properties': {'a_boolean': {'readonly': False, 'type': ['boolean', 'null']}}}}, 'required': ['versioned_object.namespace', 'versioned_object.name', 'versioned_object.version', 'versioned_object.data']}, schema)
    jsonschema.validate(self.FakeObject(a_boolean=True).obj_to_primitive(), self.FakeObject.to_json_schema())