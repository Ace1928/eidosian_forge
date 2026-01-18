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
def test_resource_type_association_create_non_visible_namesp_admin(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT2, roles=['admin'])
    rt = resource_types.ResourceTypeAssociation()
    rt.name = RESOURCE_TYPE2
    rt.prefix = 'pref'
    rt = self.rt_controller.create(request, rt, NAMESPACE1)
    self.assertEqual(RESOURCE_TYPE2, rt.name)
    self.assertEqual('pref', rt.prefix)
    self.assertNotificationLog('metadef_resource_type.create', [{'name': RESOURCE_TYPE2, 'namespace': NAMESPACE1}])
    output = self.rt_controller.show(request, NAMESPACE1)
    self.assertEqual(2, len(output.resource_type_associations))
    actual = set([x.name for x in output.resource_type_associations])
    expected = set([RESOURCE_TYPE1, RESOURCE_TYPE2])
    self.assertEqual(expected, actual)