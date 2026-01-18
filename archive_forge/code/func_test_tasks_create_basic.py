from unittest import mock
import oslo_policy.policy
from oslo_serialization import jsonutils
from glance.api import policy
from glance.tests import functional
def test_tasks_create_basic(self):
    self.start_server()
    path = '/v2/tasks'
    task = self._create_task(path=path, data=TASK1)
    self.assertEqual('import', task['type'])
    self.set_policy_rules({'tasks_api_access': '!'})
    resp = self.api_post(path, json=TASK2)
    self.assertEqual(403, resp.status_code)