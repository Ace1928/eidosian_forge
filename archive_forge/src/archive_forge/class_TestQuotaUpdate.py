from troveclient.osc.v1 import database_quota
from troveclient.tests.osc.v1 import fakes
class TestQuotaUpdate(TestQuota):

    def setUp(self):
        super(TestQuotaUpdate, self).setUp()
        self.cmd = database_quota.UpdateDatabaseQuota(self.app, None)
        self.data = self.fake_quota.fake_instances_quota
        self.quota_client.update.return_value = self.data

    def test_update_quota(self):
        args = ['tenant_id', 'instances', '51']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(('instances',), columns)
        self.assertEqual((51,), data)