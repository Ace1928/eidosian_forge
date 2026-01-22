from unittest import mock
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_volume
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
class TestServerVolumeUpdate(compute_fakes.TestComputev2):

    def setUp(self):
        super().setUp()
        self.server = compute_fakes.create_one_sdk_server()
        self.compute_sdk_client.find_server.return_value = self.server
        self.volume = volume_fakes.create_one_sdk_volume()
        self.volume_sdk_client.find_volume.return_value = self.volume
        self.cmd = server_volume.UpdateServerVolume(self.app, None)

    def test_server_volume_update(self):
        arglist = [self.server.id, self.volume.id]
        verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.update_volume_attachment.assert_not_called()
        self.assertIsNone(result)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_volume_update_with_delete_on_termination(self, sm_mock):
        sm_mock.return_value = True
        arglist = [self.server.id, self.volume.id, '--delete-on-termination']
        verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.update_volume_attachment.assert_called_once_with(self.server, self.volume, delete_on_termination=True)
        self.assertIsNone(result)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_volume_update_with_preserve_on_termination(self, sm_mock):
        sm_mock.return_value = True
        arglist = [self.server.id, self.volume.id, '--preserve-on-termination']
        verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.update_volume_attachment.assert_called_once_with(self.server, self.volume, delete_on_termination=False)
        self.assertIsNone(result)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_volume_update_with_delete_on_termination_pre_v285(self, sm_mock):
        sm_mock.return_value = False
        arglist = [self.server.id, self.volume.id, '--delete-on-termination']
        verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.compute_sdk_client.update_volume_attachment.assert_not_called()

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_volume_update_with_preserve_on_termination_pre_v285(self, sm_mock):
        sm_mock.return_value = False
        arglist = [self.server.id, self.volume.id, '--preserve-on-termination']
        verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.compute_sdk_client.update_volume_attachment.assert_not_called()