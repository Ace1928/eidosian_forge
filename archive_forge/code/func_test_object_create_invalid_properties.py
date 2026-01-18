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
def test_object_create_invalid_properties(self):
    request = unit_test_utils.get_fake_request('/metadefs/namespaces/Namespace3/objects', roles=['admin'])
    body = {'name': 'My Object', 'description': 'object1 description.', 'properties': {'property1': {'type': 'integer', 'title': 'property', 'description': 'property description', 'test-key': 'test-value'}}}
    request.body = jsonutils.dump_as_bytes(body)
    self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)