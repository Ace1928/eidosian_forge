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
def test_tag_update_name(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    tag = self.tag_controller.show(request, NAMESPACE1, TAG1)
    tag.name = TAG2
    tag = self.tag_controller.update(request, tag, NAMESPACE1, TAG1)
    self.assertEqual(TAG2, tag.name)
    self.assertNotificationLog('metadef_tag.update', [{'name': TAG2, 'name_old': TAG1, 'namespace': NAMESPACE1}])
    tag = self.tag_controller.show(request, NAMESPACE1, TAG2)
    self.assertEqual(TAG2, tag.name)