from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
class TestVolumeMessageShow(TestVolumeMessage):
    fake_message = volume_fakes.create_one_volume_message()
    columns = ('created_at', 'event_id', 'guaranteed_until', 'id', 'message_level', 'request_id', 'resource_type', 'resource_uuid', 'user_message')
    data = (fake_message.created_at, fake_message.event_id, fake_message.guaranteed_until, fake_message.id, fake_message.message_level, fake_message.request_id, fake_message.resource_type, fake_message.resource_uuid, fake_message.user_message)

    def setUp(self):
        super().setUp()
        self.volume_messages_mock.get.return_value = self.fake_message
        self.cmd = volume_message.ShowMessage(self.app, None)

    def test_message_show(self):
        self.volume_client.api_version = api_versions.APIVersion('3.3')
        arglist = [self.fake_message.id]
        verifylist = [('message_id', self.fake_message.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_messages_mock.get.assert_called_with(self.fake_message.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_message_show_pre_v33(self):
        self.volume_client.api_version = api_versions.APIVersion('3.2')
        arglist = [self.fake_message.id]
        verifylist = [('message_id', self.fake_message.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.3 or greater is required', str(exc))