import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
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