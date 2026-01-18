from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import notificationtypes
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_notification_types_list(self, mc):
    mc.return_value = c = FakeV2Client()
    c.notificationtypes.list.return_value = [{'type': 'WEBHOOK'}, {'type': 'EMAIL'}, {'type': 'PAGERDUTY'}]
    raw_args = []
    name, cmd_clazz = migr.create_command_class('do_notification_type_list', shell)
    cmd = cmd_clazz(mock.Mock(), mock.Mock())
    parser = cmd.get_parser(name)
    parsed_args = parser.parse_args(raw_args)
    cmd.run(parsed_args)
    c.notificationtypes.list.assert_called_once()