from unittest import mock
import oslo_policy.policy
from oslo_serialization import jsonutils
from glance.api import policy
from glance.tests import functional
def test_tasks_get_basic(self):
    self.start_server()
    tasks = self.load_data()
    path = '/v2/tasks/%s' % tasks[0]
    task = self.api_get(path).json
    self.assertEqual('import', task['type'])
    self.set_policy_rules({'tasks_api_access': '!'})
    path = '/v2/tasks/%s' % tasks[1]
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)