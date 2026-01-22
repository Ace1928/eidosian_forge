import testtools
from unittest import mock
from aodhclient import exceptions
from aodhclient.v2 import quota_cli
class QuotaSetTest(testtools.TestCase):

    def setUp(self):
        super(QuotaSetTest, self).setUp()
        self.app = mock.Mock()
        self.quota_mgr_mock = self.app.client_manager.alarming.quota
        self.parser = mock.Mock()
        self.quota_set = quota_cli.QuotaSet(self.app, self.parser)

    def test_quota_set(self):
        self.quota_mgr_mock.create.return_value = {'project_id': 'fake_project', 'quotas': [{'limit': 20, 'resource': 'alarms'}]}
        parser = self.quota_set.get_parser('')
        args = parser.parse_args(['fake_project', '--alarm', '20'])
        ret = list(self.quota_set.take_action(args))
        self.quota_mgr_mock.create.assert_called_once_with('fake_project', [{'resource': 'alarms', 'limit': 20}])
        self.assertIn('alarms', ret[0])
        self.assertIn(20, ret[1])

    def test_quota_set_invalid_quota(self):
        parser = self.quota_set.get_parser('')
        args = parser.parse_args(['fake_project', '--alarm', '-2'])
        self.assertRaises(exceptions.CommandError, self.quota_set.take_action, args)