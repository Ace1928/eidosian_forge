from unittest import mock
from osc_lib import utils
from troveclient.osc.v1 import database_logs
from troveclient.tests.osc.v1 import fakes
class TestSetDatabaseInstanceLog(TestLogs):

    def setUp(self):
        super(TestSetDatabaseInstanceLog, self).setUp()
        self.cmd = database_logs.SetDatabaseInstanceLog(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_set_instance_log(self, mock_find):
        mock_find.return_value = 'fake_instance_id'
        data = self.fake_logs.get_logs()[0]
        data.status = 'Ready'
        self.instance_client.log_action.return_value = data
        args = ['instance1', 'log_name', '--enable']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.instance_client.log_action.assert_called_once_with('fake_instance_id', 'log_name', enable=True, disable=False, discard=False, publish=False)