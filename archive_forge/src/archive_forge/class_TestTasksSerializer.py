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
class TestTasksSerializer(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTasksSerializer, self).setUp()
        self.serializer = glance.api.v2.tasks.ResponseSerializer()
        self.fixtures = [_domain_fixture(UUID1, type='import', status='pending', task_input={'loc': 'fake'}, result={}, owner=TENANT1, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id', message='', created_at=DATETIME, updated_at=DATETIME), _domain_fixture(UUID2, type='import', status='processing', task_input={'loc': 'bake'}, owner=TENANT2, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id', message='', created_at=DATETIME, updated_at=DATETIME, result={}), _domain_fixture(UUID3, type='import', status='success', task_input={'loc': 'foo'}, owner=TENANT3, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id', message='', created_at=DATETIME, updated_at=DATETIME, result={}, expires_at=DATETIME), _domain_fixture(UUID4, type='import', status='failure', task_input={'loc': 'boo'}, owner=TENANT4, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id', message='', created_at=DATETIME, updated_at=DATETIME, result={}, expires_at=DATETIME)]

    def test_index(self):
        expected = {'tasks': [{'id': UUID1, 'type': 'import', 'status': 'pending', 'owner': TENANT1, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID1, 'schema': '/v2/schemas/task'}, {'id': UUID2, 'type': 'import', 'status': 'processing', 'owner': TENANT2, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID2, 'schema': '/v2/schemas/task'}, {'id': UUID3, 'type': 'import', 'status': 'success', 'owner': TENANT3, 'expires_at': ISOTIME, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID3, 'schema': '/v2/schemas/task'}, {'id': UUID4, 'type': 'import', 'status': 'failure', 'owner': TENANT4, 'expires_at': ISOTIME, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID4, 'schema': '/v2/schemas/task'}], 'first': '/v2/tasks', 'schema': '/v2/schemas/tasks'}
        request = webob.Request.blank('/v2/tasks')
        response = webob.Response(request=request)
        task_fixtures = [f for f in self.fixtures]
        result = {'tasks': task_fixtures}
        self.serializer.index(response, result)
        actual = jsonutils.loads(response.body)
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)

    def test_index_next_marker(self):
        request = webob.Request.blank('/v2/tasks')
        response = webob.Response(request=request)
        task_fixtures = [f for f in self.fixtures]
        result = {'tasks': task_fixtures, 'next_marker': UUID2}
        self.serializer.index(response, result)
        output = jsonutils.loads(response.body)
        self.assertEqual('/v2/tasks?marker=%s' % UUID2, output['next'])

    def test_index_carries_query_parameters(self):
        url = '/v2/tasks?limit=10&sort_key=id&sort_dir=asc'
        request = webob.Request.blank(url)
        response = webob.Response(request=request)
        task_fixtures = [f for f in self.fixtures]
        result = {'tasks': task_fixtures, 'next_marker': UUID2}
        self.serializer.index(response, result)
        output = jsonutils.loads(response.body)
        expected_url = '/v2/tasks?limit=10&sort_dir=asc&sort_key=id'
        self.assertEqual(unit_test_utils.sort_url_by_qs_keys(expected_url), unit_test_utils.sort_url_by_qs_keys(output['first']))
        expect_next = '/v2/tasks?limit=10&marker=%s&sort_dir=asc&sort_key=id'
        self.assertEqual(unit_test_utils.sort_url_by_qs_keys(expect_next % UUID2), unit_test_utils.sort_url_by_qs_keys(output['next']))

    def test_get(self):
        expected = {'id': UUID4, 'type': 'import', 'status': 'failure', 'input': {'loc': 'boo'}, 'result': {}, 'owner': TENANT4, 'message': '', 'created_at': ISOTIME, 'updated_at': ISOTIME, 'expires_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID4, 'schema': '/v2/schemas/task', 'image_id': 'fake_image_id', 'user_id': 'fake_user', 'request_id': 'fake_request_id'}
        response = webob.Response()
        self.serializer.get(response, self.fixtures[3])
        actual = jsonutils.loads(response.body)
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)

    def test_get_ensure_expires_at_not_returned(self):
        expected = {'id': UUID1, 'type': 'import', 'status': 'pending', 'input': {'loc': 'fake'}, 'result': {}, 'owner': TENANT1, 'message': '', 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID1, 'schema': '/v2/schemas/task', 'image_id': 'fake_image_id', 'user_id': 'fake_user', 'request_id': 'fake_request_id'}
        response = webob.Response()
        self.serializer.get(response, self.fixtures[0])
        actual = jsonutils.loads(response.body)
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)
        expected = {'id': UUID2, 'type': 'import', 'status': 'processing', 'input': {'loc': 'bake'}, 'result': {}, 'owner': TENANT2, 'message': '', 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID2, 'schema': '/v2/schemas/task', 'image_id': 'fake_image_id', 'user_id': 'fake_user', 'request_id': 'fake_request_id'}
        response = webob.Response()
        self.serializer.get(response, self.fixtures[1])
        actual = jsonutils.loads(response.body)
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)

    def test_create(self):
        response = webob.Response()
        self.serializer.create(response, self.fixtures[3])
        serialized_task = jsonutils.loads(response.body)
        self.assertEqual(http.CREATED, response.status_int)
        self.assertEqual(self.fixtures[3].task_id, serialized_task['id'])
        self.assertEqual(self.fixtures[3].task_input, serialized_task['input'])
        self.assertIn('expires_at', serialized_task)
        self.assertEqual('application/json', response.content_type)

    def test_create_ensure_expires_at_is_not_returned(self):
        response = webob.Response()
        self.serializer.create(response, self.fixtures[0])
        serialized_task = jsonutils.loads(response.body)
        self.assertEqual(http.CREATED, response.status_int)
        self.assertEqual(self.fixtures[0].task_id, serialized_task['id'])
        self.assertEqual(self.fixtures[0].task_input, serialized_task['input'])
        self.assertNotIn('expires_at', serialized_task)
        self.assertEqual('application/json', response.content_type)
        response = webob.Response()
        self.serializer.create(response, self.fixtures[1])
        serialized_task = jsonutils.loads(response.body)
        self.assertEqual(http.CREATED, response.status_int)
        self.assertEqual(self.fixtures[1].task_id, serialized_task['id'])
        self.assertEqual(self.fixtures[1].task_input, serialized_task['input'])
        self.assertNotIn('expires_at', serialized_task)
        self.assertEqual('application/json', response.content_type)