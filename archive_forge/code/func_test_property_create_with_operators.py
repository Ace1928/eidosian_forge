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
def test_property_create_with_operators(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    property = properties.PropertyType()
    property.name = PROPERTY2
    property.type = 'string'
    property.title = 'title'
    property.operators = ['<or>']
    property = self.property_controller.create(request, NAMESPACE1, property)
    self.assertEqual(PROPERTY2, property.name)
    self.assertEqual('string', property.type)
    self.assertEqual('title', property.title)
    self.assertEqual(['<or>'], property.operators)
    property = self.property_controller.show(request, NAMESPACE1, PROPERTY2)
    self.assertEqual(PROPERTY2, property.name)
    self.assertEqual('string', property.type)
    self.assertEqual('title', property.title)
    self.assertEqual(['<or>'], property.operators)