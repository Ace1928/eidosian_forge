from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import notifications
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_notification_create_email(self, mc):
    mc.return_value = c = FakeV2Client()
    raw_args = ['email1', 'EMAIL', 'john.doe@hp.com']
    name, cmd_clazz = migr.create_command_class('do_notification_create', shell)
    cmd = cmd_clazz(mock.Mock(), mock.Mock())
    parser = cmd.get_parser(name)
    parsed_args = parser.parse_args(raw_args)
    cmd.run(parsed_args)
    data = {'name': 'email1', 'type': 'EMAIL', 'address': 'john.doe@hp.com'}
    c.notifications.create.assert_called_once_with(**data)