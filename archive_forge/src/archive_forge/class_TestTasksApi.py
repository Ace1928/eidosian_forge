import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
class TestTasksApi(base.ApiTest):

    def __init__(self, *args, **kwargs):
        super(TestTasksApi, self).__init__(*args, **kwargs)
        self.api_flavor = 'fakeauth'

    def _wait_on_task_execution(self, max_wait=5):
        """Wait until all the tasks have finished execution and are in
        state of success or failure.
        """
        start = timeutils.utcnow()
        while timeutils.delta_seconds(start, timeutils.utcnow()) < max_wait:
            wait = False
            path = '/v2/tasks'
            res, content = self.http.request(path, 'GET', headers=minimal_task_headers())
            content_dict = json.loads(content)
            self.assertEqual(http.client.OK, res.status)
            res_tasks = content_dict['tasks']
            if len(res_tasks) != 0:
                for task in res_tasks:
                    if task['status'] in ('pending', 'processing'):
                        wait = True
                        break
            if wait:
                eventlet.sleep(0.05)
                continue
            else:
                break

    def _post_new_task(self, **kwargs):
        task_owner = kwargs.get('owner')
        headers = minimal_task_headers(task_owner)
        task_data = _new_task_fixture()
        task_data['input']['import_from'] = 'http://example.com'
        body_content = json.dumps(task_data)
        path = '/v2/tasks'
        response, content = self.http.request(path, 'POST', headers=headers, body=body_content)
        self.assertEqual(http.client.CREATED, response.status)
        task = json.loads(content)
        task_id = task['id']
        self.assertIsNotNone(task_id)
        self.assertEqual(task_owner, task['owner'])
        self.assertEqual(task_data['type'], task['type'])
        self.assertEqual(task_data['input'], task['input'])
        self.assertEqual('http://localhost' + path + '/' + task_id, response.webob_resp.headers['Location'])
        return (task, task_data)

    def test_all_task_api(self):
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        content_dict = json.loads(content)
        self.assertEqual(http.client.OK, response.status)
        self.assertFalse(content_dict['tasks'])
        task_id = 'NON_EXISTENT_TASK'
        path = '/v2/tasks/%s' % task_id
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.NOT_FOUND, response.status)
        task_owner = 'tenant1'
        data, req_input = self._post_new_task(owner=task_owner)
        task_id = data['id']
        path = '/v2/tasks/%s' % task_id
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        self._wait_on_task_execution(max_wait=10)
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        self.assertIsNotNone(content)
        data = json.loads(content)
        self.assertIsNotNone(data)
        self.assertEqual(1, len(data['tasks']))
        expected_keys = set(['id', 'expires_at', 'type', 'owner', 'status', 'created_at', 'updated_at', 'self', 'schema'])
        task = data['tasks'][0]
        self.assertEqual(expected_keys, set(task.keys()))
        self.assertEqual(req_input['type'], task['type'])
        self.assertEqual(task_owner, task['owner'])
        self.assertEqual('success', task['status'])
        self.assertIsNotNone(task['created_at'])
        self.assertIsNotNone(task['updated_at'])

    def test_task_schema_api(self):
        path = '/v2/schemas/task'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        schema = tasks.get_task_schema()
        expected_schema = schema.minimal()
        data = json.loads(content)
        self.assertIsNotNone(data)
        self.assertEqual(expected_schema, data)
        path = '/v2/schemas/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        schema = tasks.get_collection_schema()
        expected_schema = schema.minimal()
        data = json.loads(content)
        self.assertIsNotNone(data)
        self.assertEqual(expected_schema, data)
        self._wait_on_task_execution()

    def test_create_new_task(self):
        task_data = _new_task_fixture()
        task_owner = 'tenant1'
        body_content = json.dumps(task_data)
        path = '/v2/tasks'
        response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
        self.assertEqual(http.client.CREATED, response.status)
        data = json.loads(content)
        task_id = data['id']
        self.assertIsNotNone(task_id)
        self.assertEqual(task_owner, data['owner'])
        self.assertEqual(task_data['type'], data['type'])
        self.assertEqual(task_data['input'], data['input'])
        task_data = _new_task_fixture(type='invalid')
        task_owner = 'tenant1'
        body_content = json.dumps(task_data)
        path = '/v2/tasks'
        response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
        self.assertEqual(http.client.BAD_REQUEST, response.status)
        task_data = _new_task_fixture(task_input='{something: invalid}')
        task_owner = 'tenant1'
        body_content = json.dumps(task_data)
        path = '/v2/tasks'
        response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
        self.assertEqual(http.client.BAD_REQUEST, response.status)
        self._wait_on_task_execution()

    def test_tasks_with_filter(self):
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        content_dict = json.loads(content)
        self.assertFalse(content_dict['tasks'])
        task_ids = []
        task_owner = TENANT1
        data, req_input1 = self._post_new_task(owner=task_owner)
        task_ids.append(data['id'])
        task_owner = TENANT2
        data, req_input2 = self._post_new_task(owner=task_owner)
        task_ids.append(data['id'])
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        content_dict = json.loads(content)
        self.assertEqual(2, len(content_dict['tasks']))
        params = 'owner=%s' % TENANT1
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        content_dict = json.loads(content)
        self.assertEqual(1, len(content_dict['tasks']))
        self.assertEqual(TENANT1, content_dict['tasks'][0]['owner'])
        params = 'owner=%s' % TENANT2
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        content_dict = json.loads(content)
        self.assertEqual(1, len(content_dict['tasks']))
        self.assertEqual(TENANT2, content_dict['tasks'][0]['owner'])
        params = 'type=import'
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        content_dict = json.loads(content)
        self.assertEqual(2, len(content_dict['tasks']))
        actual_task_ids = [task['id'] for task in content_dict['tasks']]
        self.assertEqual(set(task_ids), set(actual_task_ids))
        self._wait_on_task_execution()

    def test_limited_tasks(self):
        """
        Ensure marker and limit query params work
        """
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        tasks = json.loads(content)
        self.assertFalse(tasks['tasks'])
        task_ids = []
        task, _ = self._post_new_task(owner=TENANT1)
        task_ids.append(task['id'])
        task, _ = self._post_new_task(owner=TENANT2)
        task_ids.append(task['id'])
        task, _ = self._post_new_task(owner=TENANT3)
        task_ids.append(task['id'])
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        tasks = json.loads(content)['tasks']
        self.assertEqual(3, len(tasks))
        params = 'limit=2'
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(2, len(actual_tasks))
        self.assertEqual(tasks[0]['id'], actual_tasks[0]['id'])
        self.assertEqual(tasks[1]['id'], actual_tasks[1]['id'])
        params = 'marker=%s' % tasks[0]['id']
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(2, len(actual_tasks))
        self.assertEqual(tasks[1]['id'], actual_tasks[0]['id'])
        self.assertEqual(tasks[2]['id'], actual_tasks[1]['id'])
        params = 'limit=1&marker=%s' % tasks[1]['id']
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(1, len(actual_tasks))
        self.assertEqual(tasks[2]['id'], actual_tasks[0]['id'])
        self._wait_on_task_execution()

    def test_ordered_tasks(self):
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        tasks = json.loads(content)
        self.assertFalse(tasks['tasks'])
        task_ids = []
        task, _ = self._post_new_task(owner=TENANT1)
        task_ids.append(task['id'])
        task, _ = self._post_new_task(owner=TENANT2)
        task_ids.append(task['id'])
        task, _ = self._post_new_task(owner=TENANT3)
        task_ids.append(task['id'])
        path = '/v2/tasks'
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(3, len(actual_tasks))
        self.assertEqual(task_ids[2], actual_tasks[0]['id'])
        self.assertEqual(task_ids[1], actual_tasks[1]['id'])
        self.assertEqual(task_ids[0], actual_tasks[2]['id'])
        params = 'sort_key=owner&sort_dir=asc'
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        expected_task_owners = [TENANT1, TENANT2, TENANT3]
        expected_task_owners.sort()
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(3, len(actual_tasks))
        self.assertEqual(expected_task_owners, [t['owner'] for t in actual_tasks])
        params = 'sort_key=owner&sort_dir=desc&marker=%s' % task_ids[0]
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(2, len(actual_tasks))
        self.assertEqual(task_ids[2], actual_tasks[0]['id'])
        self.assertEqual(task_ids[1], actual_tasks[1]['id'])
        self.assertEqual(TENANT3, actual_tasks[0]['owner'])
        self.assertEqual(TENANT2, actual_tasks[1]['owner'])
        params = 'sort_key=owner&sort_dir=asc&marker=%s' % task_ids[0]
        path = '/v2/tasks?%s' % params
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        actual_tasks = json.loads(content)['tasks']
        self.assertEqual(0, len(actual_tasks))
        self._wait_on_task_execution()

    def test_delete_task(self):
        task_data = _new_task_fixture()
        task_owner = 'tenant1'
        body_content = json.dumps(task_data)
        path = '/v2/tasks'
        response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
        self.assertEqual(http.client.CREATED, response.status)
        data = json.loads(content)
        task_id = data['id']
        path = '/v2/tasks/%s' % task_id
        response, content = self.http.request(path, 'DELETE', headers=minimal_task_headers())
        self.assertEqual(http.client.METHOD_NOT_ALLOWED, response.status)
        self.assertEqual('GET', response.webob_resp.headers.get('Allow'))
        self.assertEqual(('GET',), response.webob_resp.allow)
        self.assertEqual(('GET',), response.allow)
        path = '/v2/tasks/%s' % task_id
        response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
        self.assertEqual(http.client.OK, response.status)
        self.assertIsNotNone(content)
        self._wait_on_task_execution()