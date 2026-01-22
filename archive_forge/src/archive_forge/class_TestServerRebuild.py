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
class TestServerRebuild(TestServer):

    def setUp(self):
        super(TestServerRebuild, self).setUp()
        self.image = image_fakes.create_one_image()
        self.image_client.get_image.return_value = self.image
        attrs = {'image': {'id': self.image.id}, 'networks': {}, 'adminPass': 'passw0rd'}
        new_server = compute_fakes.create_one_server(attrs=attrs)
        attrs['id'] = new_server.id
        attrs['status'] = 'ACTIVE'
        methods = {'rebuild': new_server}
        self.server = compute_fakes.create_one_server(attrs=attrs, methods=methods)
        self.servers_mock.get.return_value = self.server
        self.cmd = server.RebuildServer(self.app, None)

    def test_rebuild_with_image_name(self):
        image_name = 'my-custom-image'
        user_image = image_fakes.create_one_image(attrs={'name': image_name})
        self.image_client.find_image.return_value = user_image
        attrs = {'image': {'id': user_image.id}, 'networks': {}, 'adminPass': 'passw0rd'}
        new_server = compute_fakes.create_one_server(attrs=attrs)
        self.server.rebuild.return_value = new_server
        arglist = [self.server.id, '--image', image_name]
        verifylist = [('server', self.server.id), ('image', image_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.find_image.assert_called_with(image_name, ignore_missing=False)
        self.image_client.get_image.assert_called_with(user_image.id)
        self.server.rebuild.assert_called_with(user_image, None)

    def test_rebuild_with_current_image(self):
        arglist = [self.server.id]
        verifylist = [('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.find_image.assert_not_called()
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None)

    def test_rebuild_with_volume_backed_server_no_image(self):
        self.server.image = ''
        arglist = [self.server.id]
        verifylist = [('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('The --image option is required', str(exc))

    def test_rebuild_with_name(self):
        name = 'test-server-xxx'
        arglist = [self.server.id, '--name', name]
        verifylist = [('server', self.server.id), ('name', name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, name=name)

    def test_rebuild_with_preserve_ephemeral(self):
        arglist = [self.server.id, '--preserve-ephemeral']
        verifylist = [('server', self.server.id), ('preserve_ephemeral', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, preserve_ephemeral=True)

    def test_rebuild_with_no_preserve_ephemeral(self):
        arglist = [self.server.id, '--no-preserve-ephemeral']
        verifylist = [('server', self.server.id), ('preserve_ephemeral', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, preserve_ephemeral=False)

    def test_rebuild_with_password(self):
        password = 'password-xxx'
        arglist = [self.server.id, '--password', password]
        verifylist = [('server', self.server.id), ('password', password)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, password)

    def test_rebuild_with_description(self):
        self.compute_client.api_version = api_versions.APIVersion('2.19')
        description = 'description1'
        arglist = [self.server.id, '--description', description]
        verifylist = [('server', self.server.id), ('description', description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, description=description)

    def test_rebuild_with_description_pre_v219(self):
        self.compute_client.api_version = api_versions.APIVersion('2.18')
        description = 'description1'
        arglist = [self.server.id, '--description', description]
        verifylist = [('server', self.server.id), ('description', description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_rebuild_with_wait_ok(self, mock_wait_for_status):
        arglist = ['--wait', self.server.id]
        verifylist = [('wait', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, success_status=['active'])
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=False)
    def test_rebuild_with_wait_fails(self, mock_wait_for_status):
        arglist = ['--wait', self.server.id]
        verifylist = [('wait', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, success_status=['active'])
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_rebuild_with_wait_shutoff_status(self, mock_wait_for_status):
        self.server.status = 'SHUTOFF'
        arglist = ['--wait', self.server.id]
        verifylist = [('wait', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, success_status=['shutoff'])
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_rebuild_with_wait_error_status(self, mock_wait_for_status):
        self.server.status = 'ERROR'
        arglist = ['--wait', self.server.id]
        verifylist = [('wait', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, success_status=['active'])
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None)

    def test_rebuild_wrong_status_fails(self):
        self.server.status = 'SHELVED'
        arglist = [self.server.id]
        verifylist = [('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_not_called()

    def test_rebuild_with_property(self):
        arglist = [self.server.id, '--property', 'key1=value1', '--property', 'key2=value2']
        expected_properties = {'key1': 'value1', 'key2': 'value2'}
        verifylist = [('server', self.server.id), ('properties', expected_properties)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, meta=expected_properties)

    def test_rebuild_with_keypair_name(self):
        self.compute_client.api_version = api_versions.APIVersion('2.54')
        self.server.key_name = 'mykey'
        arglist = [self.server.id, '--key-name', self.server.key_name]
        verifylist = [('server', self.server.id), ('key_name', self.server.key_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, key_name=self.server.key_name)

    def test_rebuild_with_keypair_name_pre_v254(self):
        self.compute_client.api_version = api_versions.APIVersion('2.53')
        self.server.key_name = 'mykey'
        arglist = [self.server.id, '--key-name', self.server.key_name]
        verifylist = [('server', self.server.id), ('key_name', self.server.key_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_rebuild_with_no_keypair_name(self):
        self.compute_client.api_version = api_versions.APIVersion('2.54')
        self.server.key_name = 'mykey'
        arglist = [self.server.id, '--no-key-name']
        verifylist = [('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, key_name=None)

    def test_rebuild_with_keypair_name_and_unset(self):
        self.server.key_name = 'mykey'
        arglist = [self.server.id, '--key-name', self.server.key_name, '--no-key-name']
        verifylist = [('server', self.server.id), ('key_name', self.server.key_name)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    @mock.patch('openstackclient.compute.v2.server.io.open')
    def test_rebuild_with_user_data(self, mock_open):
        self.compute_client.api_version = api_versions.APIVersion('2.57')
        mock_file = mock.Mock(name='File')
        mock_open.return_value = mock_file
        mock_open.read.return_value = '#!/bin/sh'
        arglist = [self.server.id, '--user-data', 'userdata.sh']
        verifylist = [('server', self.server.id), ('user_data', 'userdata.sh')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock_open.assert_called_with('userdata.sh')
        mock_file.close.assert_called_with()
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, userdata=mock_file)

    def test_rebuild_with_user_data_pre_v257(self):
        self.compute_client.api_version = api_versions.APIVersion('2.56')
        arglist = [self.server.id, '--user-data', 'userdata.sh']
        verifylist = [('server', self.server.id), ('user_data', 'userdata.sh')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_rebuild_with_no_user_data(self):
        self.compute_client.api_version = api_versions.APIVersion('2.54')
        self.server.key_name = 'mykey'
        arglist = [self.server.id, '--no-user-data']
        verifylist = [('server', self.server.id), ('no_user_data', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, userdata=None)

    def test_rebuild_with_no_user_data_pre_v254(self):
        self.compute_client.api_version = api_versions.APIVersion('2.53')
        arglist = [self.server.id, '--no-user-data']
        verifylist = [('server', self.server.id), ('no_user_data', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_rebuild_with_user_data_and_unset(self):
        arglist = [self.server.id, '--user-data', 'userdata.sh', '--no-user-data']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_rebuild_with_trusted_image_cert(self):
        self.compute_client.api_version = api_versions.APIVersion('2.63')
        arglist = [self.server.id, '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar']
        verifylist = [('server', self.server.id), ('trusted_image_certs', ['foo', 'bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, trusted_image_certificates=['foo', 'bar'])

    def test_rebuild_with_trusted_image_cert_pre_v263(self):
        self.compute_client.api_version = api_versions.APIVersion('2.62')
        arglist = [self.server.id, '--trusted-image-cert', 'foo', '--trusted-image-cert', 'bar']
        verifylist = [('server', self.server.id), ('trusted_image_certs', ['foo', 'bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_rebuild_with_no_trusted_image_cert(self):
        self.compute_client.api_version = api_versions.APIVersion('2.63')
        arglist = [self.server.id, '--no-trusted-image-certs']
        verifylist = [('server', self.server.id), ('no_trusted_image_certs', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, trusted_image_certificates=None)

    def test_rebuild_with_no_trusted_image_cert_pre_v263(self):
        self.compute_client.api_version = api_versions.APIVersion('2.62')
        arglist = [self.server.id, '--no-trusted-image-certs']
        verifylist = [('server', self.server.id), ('no_trusted_image_certs', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_rebuild_with_hostname(self):
        self.compute_client.api_version = api_versions.APIVersion('2.90')
        arglist = [self.server.id, '--hostname', 'new-hostname']
        verifylist = [('server', self.server.id), ('hostname', 'new-hostname')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.get_image.assert_called_with(self.image.id)
        self.server.rebuild.assert_called_with(self.image, None, hostname='new-hostname')

    def test_rebuild_with_hostname_pre_v290(self):
        self.compute_client.api_version = api_versions.APIVersion('2.89')
        arglist = [self.server.id, '--hostname', 'new-hostname']
        verifylist = [('server', self.server.id), ('hostname', 'new-hostname')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)