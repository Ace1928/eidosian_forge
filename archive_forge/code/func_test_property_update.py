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
def test_property_update(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT3, roles=['admin'])
    property = self.property_controller.show(request, NAMESPACE3, PROPERTY1)
    property.name = PROPERTY1
    property.type = 'string123'
    property.title = 'title123'
    property = self.property_controller.update(request, NAMESPACE3, PROPERTY1, property)
    self.assertEqual(PROPERTY1, property.name)
    self.assertEqual('string123', property.type)
    self.assertEqual('title123', property.title)
    self.assertNotificationLog('metadef_property.update', [{'name': PROPERTY1, 'namespace': NAMESPACE3, 'type': 'string123', 'title': 'title123'}])
    property = self.property_controller.show(request, NAMESPACE3, PROPERTY1)
    self.assertEqual(PROPERTY1, property.name)
    self.assertEqual('string123', property.type)
    self.assertEqual('title123', property.title)