import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_object_create_overlimit_name(self):
    request = unit_test_utils.get_fake_request('/metadefs/namespaces/Namespace3/objects', roles=['admin'])
    request.body = jsonutils.dump_as_bytes({'name': 'a' * 81})
    exc = self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)
    self.assertIn("Failed validating 'maxLength' in schema['properties']['name']", exc.explanation)