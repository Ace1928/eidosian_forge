from unittest import mock
from osc_lib import utils
from troveclient.osc.v1 import database_logs
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_show_instance_log(self, mock_find):
    mock_find.return_value = 'fake_instance_id'
    data = self.fake_logs.get_logs()[0]
    self.instance_client.log_show.return_value = data
    args = ['instance', 'logname']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, values = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(data.to_dict().values(), values)