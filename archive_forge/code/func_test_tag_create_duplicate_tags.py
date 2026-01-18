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
def test_tag_create_duplicate_tags(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    metadef_tags = tags.MetadefTags()
    metadef_tags.tags = _db_tags_fixture([TAG4, TAG5, TAG4])
    self.assertRaises(webob.exc.HTTPConflict, self.tag_controller.create_tags, request, metadef_tags, NAMESPACE1)
    self.assertNotificationsLog([])