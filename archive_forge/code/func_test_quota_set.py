import testtools
from unittest import mock
from aodhclient import exceptions
from aodhclient.v2 import quota_cli
def test_quota_set(self):
    self.quota_mgr_mock.create.return_value = {'project_id': 'fake_project', 'quotas': [{'limit': 20, 'resource': 'alarms'}]}
    parser = self.quota_set.get_parser('')
    args = parser.parse_args(['fake_project', '--alarm', '20'])
    ret = list(self.quota_set.take_action(args))
    self.quota_mgr_mock.create.assert_called_once_with('fake_project', [{'resource': 'alarms', 'limit': 20}])
    self.assertIn('alarms', ret[0])
    self.assertIn(20, ret[1])