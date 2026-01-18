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
def test_resource_type_association_delete_non_visible(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT3)
    self.assertRaises(webob.exc.HTTPNotFound, self.rt_controller.delete, request, NAMESPACE1, RESOURCE_TYPE1)
    self.assertNotificationsLog([])