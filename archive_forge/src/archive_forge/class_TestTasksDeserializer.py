import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestTasksDeserializer(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTasksDeserializer, self).setUp()
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.deserializer = glance.api.v2.tasks.RequestDeserializer(policy_engine=self.policy)

    def test_create_no_body(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_create(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'type': 'import', 'input': {'import_from': 'swift://cloud.foo/myaccount/mycontainer/path', 'import_from_format': 'qcow2', 'image_properties': {'name': 'fake1'}}})
        output = self.deserializer.create(request)
        properties = {'type': 'import', 'input': {'import_from': 'swift://cloud.foo/myaccount/mycontainer/path', 'import_from_format': 'qcow2', 'image_properties': {'name': 'fake1'}}}
        self.maxDiff = None
        expected = {'task': properties}
        self.assertEqual(expected, output)

    def test_index(self):
        marker = str(uuid.uuid4())
        path = '/tasks?limit=1&marker=%s' % marker
        request = unit_test_utils.get_fake_request(path)
        expected = {'limit': 1, 'marker': marker, 'sort_key': 'created_at', 'sort_dir': 'desc', 'filters': {}}
        output = self.deserializer.index(request)
        self.assertEqual(expected, output)

    def test_index_strip_params_from_filters(self):
        type = 'import'
        path = '/tasks?type=%s' % type
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(type, output['filters']['type'])

    def test_index_with_many_filter(self):
        status = 'success'
        type = 'import'
        path = '/tasks?status=%(status)s&type=%(type)s' % {'status': status, 'type': type}
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(status, output['filters']['status'])
        self.assertEqual(type, output['filters']['type'])

    def test_index_with_filter_and_limit(self):
        status = 'success'
        path = '/tasks?status=%s&limit=1' % status
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(status, output['filters']['status'])
        self.assertEqual(1, output['limit'])

    def test_index_non_integer_limit(self):
        request = unit_test_utils.get_fake_request('/tasks?limit=blah')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_zero_limit(self):
        request = unit_test_utils.get_fake_request('/tasks?limit=0')
        expected = {'limit': 0, 'sort_key': 'created_at', 'sort_dir': 'desc', 'filters': {}}
        output = self.deserializer.index(request)
        self.assertEqual(expected, output)

    def test_index_negative_limit(self):
        path = '/tasks?limit=-1'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_fraction(self):
        request = unit_test_utils.get_fake_request('/tasks?limit=1.1')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_invalid_status(self):
        path = '/tasks?status=blah'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_index_marker(self):
        marker = str(uuid.uuid4())
        path = '/tasks?marker=%s' % marker
        request = unit_test_utils.get_fake_request(path)
        output = self.deserializer.index(request)
        self.assertEqual(marker, output.get('marker'))

    def test_index_marker_not_specified(self):
        request = unit_test_utils.get_fake_request('/tasks')
        output = self.deserializer.index(request)
        self.assertNotIn('marker', output)

    def test_index_limit_not_specified(self):
        request = unit_test_utils.get_fake_request('/tasks')
        output = self.deserializer.index(request)
        self.assertNotIn('limit', output)

    def test_index_sort_key_id(self):
        request = unit_test_utils.get_fake_request('/tasks?sort_key=id')
        output = self.deserializer.index(request)
        expected = {'sort_key': 'id', 'sort_dir': 'desc', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_sort_dir_asc(self):
        request = unit_test_utils.get_fake_request('/tasks?sort_dir=asc')
        output = self.deserializer.index(request)
        expected = {'sort_key': 'created_at', 'sort_dir': 'asc', 'filters': {}}
        self.assertEqual(expected, output)

    def test_index_sort_dir_bad_value(self):
        request = unit_test_utils.get_fake_request('/tasks?sort_dir=invalid')
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)