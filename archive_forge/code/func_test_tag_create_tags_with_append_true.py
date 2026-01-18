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
def test_tag_create_tags_with_append_true(self):
    request = unit_test_utils.get_fake_request(headers={'X-Openstack-Append': 'True'}, roles=['admin'])
    metadef_tags = tags.MetadefTags()
    metadef_tags.tags = _db_tags_fixture([TAG2, TAG3])
    output = self.tag_controller.create_tags(request, metadef_tags, NAMESPACE1)
    output = output.to_dict()
    self.assertEqual(2, len(output['tags']))
    actual = set([tag.name for tag in output['tags']])
    expected = set([TAG2, TAG3])
    self.assertEqual(expected, actual)
    self.assertNotificationLog('metadef_tag.create', [{'name': TAG2, 'namespace': NAMESPACE1}, {'name': TAG3, 'namespace': NAMESPACE1}])
    metadef_tags = tags.MetadefTags()
    metadef_tags.tags = _db_tags_fixture([TAG4, TAG5])
    output = self.tag_controller.create_tags(request, metadef_tags, NAMESPACE1)
    output = output.to_dict()
    self.assertEqual(2, len(output['tags']))
    actual = set([tag.name for tag in output['tags']])
    expected = set([TAG4, TAG5])
    self.assertEqual(expected, actual)
    self.assertNotificationLog('metadef_tag.create', [{'name': TAG4, 'namespace': NAMESPACE1}, {'name': TAG5, 'namespace': NAMESPACE1}])
    output = self.tag_controller.index(request, NAMESPACE1)
    output = output.to_dict()
    self.assertEqual(5, len(output['tags']))
    actual = set([tag.name for tag in output['tags']])
    expected = set([TAG1, TAG2, TAG3, TAG4, TAG5])
    self.assertEqual(expected, actual)