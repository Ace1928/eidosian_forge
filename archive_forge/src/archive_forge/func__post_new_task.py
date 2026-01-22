import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
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