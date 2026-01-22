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
class TestServerSet(TestServer):

    def setUp(self):
        super(TestServerSet, self).setUp()
        self.attrs = {'api_version': None}
        self.methods = {'update': None, 'reset_state': None, 'change_password': None, 'clear_password': None, 'add_tag': None, 'set_tags': None}
        self.fake_servers = self.setup_servers_mock(2)
        self.cmd = server.SetServer(self.app, None)

    def test_server_set_no_option(self):
        arglist = ['foo_vm']
        verifylist = [('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertNotCalled(self.fake_servers[0].update)
        self.assertNotCalled(self.fake_servers[0].reset_state)
        self.assertNotCalled(self.fake_servers[0].change_password)
        self.assertNotCalled(self.servers_mock.set_meta)
        self.assertIsNone(result)

    def test_server_set_with_state(self):
        for index, state in enumerate(['active', 'error']):
            arglist = ['--state', state, 'foo_vm']
            verifylist = [('state', state), ('server', 'foo_vm')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            result = self.cmd.take_action(parsed_args)
            self.fake_servers[index].reset_state.assert_called_once_with(state=state)
            self.assertIsNone(result)

    def test_server_set_with_invalid_state(self):
        arglist = ['--state', 'foo_state', 'foo_vm']
        verifylist = [('state', 'foo_state'), ('server', 'foo_vm')]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_server_set_with_name(self):
        arglist = ['--name', 'foo_name', 'foo_vm']
        verifylist = [('name', 'foo_name'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.fake_servers[0].update.assert_called_once_with(name='foo_name')
        self.assertIsNone(result)

    def test_server_set_with_property(self):
        arglist = ['--property', 'key1=value1', '--property', 'key2=value2', 'foo_vm']
        verifylist = [('properties', {'key1': 'value1', 'key2': 'value2'}), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.set_meta.assert_called_once_with(self.fake_servers[0], parsed_args.properties)
        self.assertIsNone(result)

    def test_server_set_with_password(self):
        arglist = ['--password', 'foo', 'foo_vm']
        verifylist = [('password', 'foo'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.fake_servers[0].change_password.assert_called_once_with('foo')

    def test_server_set_with_no_password(self):
        arglist = ['--no-password', 'foo_vm']
        verifylist = [('no_password', True), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.fake_servers[0].clear_password.assert_called_once_with()

    @mock.patch.object(getpass, 'getpass', return_value=mock.sentinel.fake_pass)
    def test_server_set_with_root_password(self, mock_getpass):
        arglist = ['--root-password', 'foo_vm']
        verifylist = [('root_password', True), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.fake_servers[0].change_password.assert_called_once_with(mock.sentinel.fake_pass)
        self.assertIsNone(result)

    def test_server_set_with_description(self):
        self.fake_servers[0].api_version = api_versions.APIVersion('2.19')
        arglist = ['--description', 'foo_description', 'foo_vm']
        verifylist = [('description', 'foo_description'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.fake_servers[0].update.assert_called_once_with(description='foo_description')
        self.assertIsNone(result)

    def test_server_set_with_description_pre_v219(self):
        self.fake_servers[0].api_version = api_versions.APIVersion('2.18')
        arglist = ['--description', 'foo_description', 'foo_vm']
        verifylist = [('description', 'foo_description'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_set_with_tag(self):
        self.fake_servers[0].api_version = api_versions.APIVersion('2.26')
        arglist = ['--tag', 'tag1', '--tag', 'tag2', 'foo_vm']
        verifylist = [('tags', ['tag1', 'tag2']), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.fake_servers[0].add_tag.assert_has_calls([mock.call(tag='tag1'), mock.call(tag='tag2')])
        self.assertIsNone(result)

    def test_server_set_with_tag_pre_v226(self):
        self.fake_servers[0].api_version = api_versions.APIVersion('2.25')
        arglist = ['--tag', 'tag1', '--tag', 'tag2', 'foo_vm']
        verifylist = [('tags', ['tag1', 'tag2']), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.26 or greater is required', str(ex))

    def test_server_set_with_hostname(self):
        self.fake_servers[0].api_version = api_versions.APIVersion('2.90')
        arglist = ['--hostname', 'foo-hostname', 'foo_vm']
        verifylist = [('hostname', 'foo-hostname'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.fake_servers[0].update.assert_called_once_with(hostname='foo-hostname')
        self.assertIsNone(result)

    def test_server_set_with_hostname_pre_v290(self):
        self.fake_servers[0].api_version = api_versions.APIVersion('2.89')
        arglist = ['--hostname', 'foo-hostname', 'foo_vm']
        verifylist = [('hostname', 'foo-hostname'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)