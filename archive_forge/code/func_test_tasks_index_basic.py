from unittest import mock
import oslo_policy.policy
from oslo_serialization import jsonutils
from glance.api import policy
from glance.tests import functional
def test_tasks_index_basic(self):
    self.start_server()
    tasks = self.load_data()
    path = '/v2/tasks'
    output = self.api_get(path).json
    self.assertEqual(len(tasks), len(output['tasks']))
    self.set_policy_rules({'tasks_api_access': '!'})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)