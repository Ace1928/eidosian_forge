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
class TestServerList(_TestServerList):

    def setUp(self):
        super(TestServerList, self).setUp()
        Image = collections.namedtuple('Image', 'id name')
        self.image_client.images.return_value = [Image(id=s.image['id'], name=self.image.name) for s in self.servers if s.image]
        Flavor = collections.namedtuple('Flavor', 'id name')
        self.compute_sdk_client.flavors.return_value = [Flavor(id=s.flavor['id'], name=self.flavor.name) for s in self.servers]
        self.data = tuple(((s.id, s.name, s.status, server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name) for s in self.servers))

    def test_server_list_no_option(self):
        arglist = []
        verifylist = [('all_projects', False), ('long', False), ('deleted', False), ('name_lookup_one_by_one', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_called()
        self.compute_sdk_client.flavors.assert_called()
        self.assertFalse(self.flavors_mock.get.call_count)
        self.assertFalse(self.image_client.get_image.call_count)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_no_servers(self):
        arglist = []
        verifylist = [('all_projects', False), ('long', False), ('deleted', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.servers.return_value = []
        self.data = ()
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_not_called()
        self.compute_sdk_client.flavors.assert_not_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_long_option(self):
        self.data = tuple(((s.id, s.name, s.status, getattr(s, 'task_state'), server.PowerStateColumn(getattr(s, 'power_state')), server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name, s.flavor['id'], getattr(s, 'availability_zone'), server.HostColumn(getattr(s, 'hypervisor_hostname')), format_columns.DictColumn(s.metadata)) for s in self.servers))
        arglist = ['--long']
        verifylist = [('all_projects', False), ('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        image_ids = {s.image['id'] for s in self.servers if s.image}
        self.image_client.images.assert_called_once_with(id=f'in:{','.join(image_ids)}')
        self.compute_sdk_client.flavors.assert_called_once_with(is_public=None)
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_column_option(self):
        arglist = ['-c', 'Project ID', '-c', 'User ID', '-c', 'Created At', '-c', 'Security Groups', '-c', 'Task State', '-c', 'Power State', '-c', 'Image ID', '-c', 'Flavor ID', '-c', 'Availability Zone', '-c', 'Host', '-c', 'Properties', '--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertIn('Project ID', columns)
        self.assertIn('User ID', columns)
        self.assertIn('Created At', columns)
        self.assertIn('Security Groups', columns)
        self.assertIn('Task State', columns)
        self.assertIn('Power State', columns)
        self.assertIn('Image ID', columns)
        self.assertIn('Flavor ID', columns)
        self.assertIn('Availability Zone', columns)
        self.assertIn('Host', columns)
        self.assertIn('Properties', columns)
        self.assertCountEqual(columns, set(columns))

    def test_server_list_no_name_lookup_option(self):
        self.data = tuple(((s.id, s.name, s.status, server.AddressesColumn(s.addresses), s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, s.flavor['id']) for s in self.servers))
        arglist = ['--no-name-lookup']
        verifylist = [('all_projects', False), ('no_name_lookup', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_not_called()
        self.compute_sdk_client.flavors.assert_not_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_n_option(self):
        self.data = tuple(((s.id, s.name, s.status, server.AddressesColumn(s.addresses), s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, s.flavor['id']) for s in self.servers))
        arglist = ['-n']
        verifylist = [('all_projects', False), ('no_name_lookup', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_not_called()
        self.compute_sdk_client.flavors.assert_not_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_name_lookup_one_by_one(self):
        arglist = ['--name-lookup-one-by-one']
        verifylist = [('all_projects', False), ('no_name_lookup', False), ('name_lookup_one_by_one', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_not_called()
        self.compute_sdk_client.flavors.assert_not_called()
        self.image_client.get_image.assert_called()
        self.compute_sdk_client.find_flavor.assert_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_with_image(self):
        arglist = ['--image', self.image.id]
        verifylist = [('image', self.image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.find_image.assert_called_with(self.image.id, ignore_missing=False)
        self.kwargs['image'] = self.image.id
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_not_called()
        self.compute_sdk_client.flavors.assert_called_once()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_with_flavor(self):
        arglist = ['--flavor', self.flavor.id]
        verifylist = [('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_has_calls([mock.call(self.flavor.id, ignore_missing=False)])
        self.kwargs['flavor'] = self.flavor.id
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.image_client.images.assert_called_once()
        self.compute_sdk_client.flavors.assert_not_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_with_changes_since(self):
        arglist = ['--changes-since', '2016-03-04T06:27:59Z', '--deleted']
        verifylist = [('changes_since', '2016-03-04T06:27:59Z'), ('deleted', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['changes-since'] = '2016-03-04T06:27:59Z'
        self.kwargs['deleted'] = True
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    @mock.patch.object(iso8601, 'parse_date', side_effect=iso8601.ParseError)
    def test_server_list_with_invalid_changes_since(self, mock_parse_isotime):
        arglist = ['--changes-since', 'Invalid time value']
        verifylist = [('changes_since', 'Invalid time value')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('Invalid changes-since value: Invalid time value', str(e))
        mock_parse_isotime.assert_called_once_with('Invalid time value')

    def test_server_list_with_tag(self):
        self._set_mock_microversion('2.26')
        arglist = ['--tag', 'tag1', '--tag', 'tag2']
        verifylist = [('tags', ['tag1', 'tag2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['tags'] = 'tag1,tag2'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_with_tag_pre_v225(self):
        self._set_mock_microversion('2.25')
        arglist = ['--tag', 'tag1', '--tag', 'tag2']
        verifylist = [('tags', ['tag1', 'tag2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.26 or greater is required', str(ex))

    def test_server_list_with_not_tag(self):
        self._set_mock_microversion('2.26')
        arglist = ['--not-tag', 'tag1', '--not-tag', 'tag2']
        verifylist = [('not_tags', ['tag1', 'tag2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['not-tags'] = 'tag1,tag2'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_server_list_with_not_tag_pre_v226(self):
        self._set_mock_microversion('2.25')
        arglist = ['--not-tag', 'tag1', '--not-tag', 'tag2']
        verifylist = [('not_tags', ['tag1', 'tag2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.26 or greater is required', str(ex))

    def test_server_list_with_availability_zone(self):
        arglist = ['--availability-zone', 'test-az']
        verifylist = [('availability_zone', 'test-az')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['availability_zone'] = 'test-az'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_key_name(self):
        arglist = ['--key-name', 'test-key']
        verifylist = [('key_name', 'test-key')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['key_name'] = 'test-key'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_config_drive(self):
        arglist = ['--config-drive']
        verifylist = [('has_config_drive', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['config_drive'] = True
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_no_config_drive(self):
        arglist = ['--no-config-drive']
        verifylist = [('has_config_drive', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['config_drive'] = False
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_progress(self):
        arglist = ['--progress', '100']
        verifylist = [('progress', 100)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['progress'] = '100'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_progress_invalid(self):
        arglist = ['--progress', '101']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verify_args=[])

    def test_server_list_with_vm_state(self):
        arglist = ['--vm-state', 'active']
        verifylist = [('vm_state', 'active')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['vm_state'] = 'active'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_task_state(self):
        arglist = ['--task-state', 'deleting']
        verifylist = [('task_state', 'deleting')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['task_state'] = 'deleting'
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_with_power_state(self):
        arglist = ['--power-state', 'running']
        verifylist = [('power_state', 'running')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['power_state'] = 1
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_list_long_with_host_status_v216(self):
        self._set_mock_microversion('2.16')
        self.data1 = tuple(((s.id, s.name, s.status, getattr(s, 'task_state'), server.PowerStateColumn(getattr(s, 'power_state')), server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name, s.flavor['id'], getattr(s, 'availability_zone'), server.HostColumn(getattr(s, 'hypervisor_hostname')), format_columns.DictColumn(s.metadata)) for s in self.servers))
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(tuple(self.data1), tuple(data))
        self.compute_sdk_client.servers.reset_mock()
        self.attrs['host_status'] = 'UP'
        servers = self.setup_sdk_servers_mock(3)
        self.compute_sdk_client.servers.return_value = servers
        Image = collections.namedtuple('Image', 'id name')
        self.image_client.images.return_value = [Image(id=s.image['id'], name=self.image.name) for s in servers if s.image]
        columns_long = self.columns_long + ('Host Status',)
        self.data2 = tuple(((s.id, s.name, s.status, getattr(s, 'task_state'), server.PowerStateColumn(getattr(s, 'power_state')), server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name, s.flavor['id'], getattr(s, 'availability_zone'), server.HostColumn(getattr(s, 'hypervisor_hostname')), format_columns.DictColumn(s.metadata), s.host_status) for s in servers))
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertEqual(columns_long, columns)
        self.assertEqual(tuple(self.data2), tuple(data))