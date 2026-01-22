import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerAddVolume(TestServerVolume):

    def setUp(self):
        super(TestServerAddVolume, self).setUp()
        self.cmd = server.AddServerVolume(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_server_add_volume(self, sm_mock):
        arglist = ['--device', '/dev/sdb', self.servers[0].id, self.volumes[0].id]
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('ID', 'Server ID', 'Volume ID', 'Device')
        expected_data = (self.volume_attachment.id, self.volume_attachment.server_id, self.volume_attachment.volume_id, '/dev/sdb')
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)
        self.compute_sdk_client.create_volume_attachment.assert_called_once_with(self.servers[0], volumeId=self.volumes[0].id, device='/dev/sdb')

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_volume_with_tag(self, sm_mock):

        def side_effect(compute_client, version):
            if version == '2.49':
                return True
            return False
        sm_mock.side_effect = side_effect
        arglist = ['--device', '/dev/sdb', '--tag', 'foo', self.servers[0].id, self.volumes[0].id]
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb'), ('tag', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('ID', 'Server ID', 'Volume ID', 'Device', 'Tag')
        expected_data = (self.volume_attachment.id, self.volume_attachment.server_id, self.volume_attachment.volume_id, self.volume_attachment.device, self.volume_attachment.tag)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)
        self.compute_sdk_client.create_volume_attachment.assert_called_once_with(self.servers[0], volumeId=self.volumes[0].id, device='/dev/sdb', tag='foo')

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_server_add_volume_with_tag_pre_v249(self, sm_mock):
        arglist = [self.servers[0].id, self.volumes[0].id, '--tag', 'foo']
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('tag', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.49 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_add_volume_with_enable_delete_on_termination(self, sm_mock):
        self.volume_attachment.delete_on_termination = True
        arglist = ['--enable-delete-on-termination', '--device', '/dev/sdb', self.servers[0].id, self.volumes[0].id]
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb'), ('enable_delete_on_termination', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('ID', 'Server ID', 'Volume ID', 'Device', 'Tag', 'Delete On Termination')
        expected_data = (self.volume_attachment.id, self.volume_attachment.server_id, self.volume_attachment.volume_id, self.volume_attachment.device, self.volume_attachment.tag, self.volume_attachment.delete_on_termination)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)
        self.compute_sdk_client.create_volume_attachment.assert_called_once_with(self.servers[0], volumeId=self.volumes[0].id, device='/dev/sdb', delete_on_termination=True)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_add_volume_with_disable_delete_on_termination(self, sm_mock):
        self.volume_attachment.delete_on_termination = False
        arglist = ['--disable-delete-on-termination', '--device', '/dev/sdb', self.servers[0].id, self.volumes[0].id]
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb'), ('disable_delete_on_termination', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('ID', 'Server ID', 'Volume ID', 'Device', 'Tag', 'Delete On Termination')
        expected_data = (self.volume_attachment.id, self.volume_attachment.server_id, self.volume_attachment.volume_id, self.volume_attachment.device, self.volume_attachment.tag, self.volume_attachment.delete_on_termination)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)
        self.compute_sdk_client.create_volume_attachment.assert_called_once_with(self.servers[0], volumeId=self.volumes[0].id, device='/dev/sdb', delete_on_termination=False)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_volume_with_enable_delete_on_termination_pre_v279(self, sm_mock):

        def side_effect(compute_client, version):
            if version == '2.79':
                return False
            return True
        sm_mock.side_effect = side_effect
        arglist = [self.servers[0].id, self.volumes[0].id, '--enable-delete-on-termination']
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('enable_delete_on_termination', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.79 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_volume_with_disable_delete_on_termination_pre_v279(self, sm_mock):

        def side_effect(compute_client, version):
            if version == '2.79':
                return False
            return True
        sm_mock.side_effect = side_effect
        arglist = [self.servers[0].id, self.volumes[0].id, '--disable-delete-on-termination']
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('disable_delete_on_termination', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.79 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_add_volume_with_disable_and_enable_delete_on_termination(self, sm_mock):
        arglist = ['--enable-delete-on-termination', '--disable-delete-on-termination', '--device', '/dev/sdb', self.servers[0].id, self.volumes[0].id]
        verifylist = [('server', self.servers[0].id), ('volume', self.volumes[0].id), ('device', '/dev/sdb'), ('enable_delete_on_termination', True), ('disable_delete_on_termination', True)]
        ex = self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertIn('argument --disable-delete-on-termination: not allowed with argument --enable-delete-on-termination', str(ex))