import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
class TestBaremetalSet(TestBaremetal):

    def setUp(self):
        super(TestBaremetalSet, self).setUp()
        self.baremetal_mock.node.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)
        self.cmd = baremetal_node.SetBaremetalNode(self.app, None)

    def test_baremetal_set_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_set_no_property(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.node.update.called)

    def test_baremetal_set_one_property(self):
        arglist = ['node_uuid', '--property', 'path/to/property=value']
        verifylist = [('nodes', ['node_uuid']), ('property', ['path/to/property=value'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/properties/path/to/property', 'value': 'value', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_multiple_properties(self):
        arglist = ['node_uuid', '--property', 'path/to/property=value', '--property', 'other/path=value2']
        verifylist = [('nodes', ['node_uuid']), ('property', ['path/to/property=value', 'other/path=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/properties/path/to/property', 'value': 'value', 'op': 'add'}, {'path': '/properties/other/path', 'value': 'value2', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_instance_uuid(self):
        arglist = ['node_uuid', '--instance-uuid', 'xxxxx']
        verifylist = [('nodes', ['node_uuid']), ('instance_uuid', 'xxxxx')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/instance_uuid', 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_name(self):
        arglist = ['node_uuid', '--name', 'xxxxx']
        verifylist = [('nodes', ['node_uuid']), ('name', 'xxxxx')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/name', 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_chassis(self):
        chassis = '4f4135ea-7e58-4e3d-bcc4-b87ca16e980b'
        arglist = ['node_uuid', '--chassis-uuid', chassis]
        verifylist = [('nodes', ['node_uuid']), ('chassis_uuid', chassis)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/chassis_uuid', 'value': chassis, 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_driver(self):
        arglist = ['node_uuid', '--driver', 'xxxxx']
        verifylist = [('nodes', ['node_uuid']), ('driver', 'xxxxx')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/driver', 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_driver_reset_interfaces(self):
        arglist = ['node_uuid', '--driver', 'xxxxx', '--reset-interfaces']
        verifylist = [('nodes', ['node_uuid']), ('driver', 'xxxxx'), ('reset_interfaces', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/driver', 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=True)

    def test_reset_interfaces_without_driver(self):
        arglist = ['node_uuid', '--reset-interfaces']
        verifylist = [('nodes', ['node_uuid']), ('reset_interfaces', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertFalse(self.baremetal_mock.node.update.called)

    def _test_baremetal_set_hardware_interface(self, interface):
        arglist = ['node_uuid', '--%s-interface' % interface, 'xxxxx']
        verifylist = [('nodes', ['node_uuid']), ('%s_interface' % interface, 'xxxxx')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/%s_interface' % interface, 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_bios_interface(self):
        self._test_baremetal_set_hardware_interface('bios')

    def test_baremetal_set_boot_interface(self):
        self._test_baremetal_set_hardware_interface('boot')

    def test_baremetal_set_console_interface(self):
        self._test_baremetal_set_hardware_interface('console')

    def test_baremetal_set_deploy_interface(self):
        self._test_baremetal_set_hardware_interface('deploy')

    def test_baremetal_set_firmware_interface(self):
        self._test_baremetal_set_hardware_interface('firmware')

    def test_baremetal_set_inspect_interface(self):
        self._test_baremetal_set_hardware_interface('inspect')

    def test_baremetal_set_management_interface(self):
        self._test_baremetal_set_hardware_interface('management')

    def test_baremetal_set_network_interface(self):
        self._test_baremetal_set_hardware_interface('network')

    def test_baremetal_set_power_interface(self):
        self._test_baremetal_set_hardware_interface('power')

    def test_baremetal_set_raid_interface(self):
        self._test_baremetal_set_hardware_interface('raid')

    def test_baremetal_set_rescue_interface(self):
        self._test_baremetal_set_hardware_interface('rescue')

    def test_baremetal_set_storage_interface(self):
        self._test_baremetal_set_hardware_interface('storage')

    def test_baremetal_set_vendor_interface(self):
        self._test_baremetal_set_hardware_interface('vendor')

    def _test_baremetal_reset_hardware_interface(self, interface):
        arglist = ['node_uuid', '--reset-%s-interface' % interface]
        verifylist = [('nodes', ['node_uuid']), ('reset_%s_interface' % interface, True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/%s_interface' % interface, 'op': 'remove'}], reset_interfaces=None)

    def test_baremetal_reset_bios_interface(self):
        self._test_baremetal_reset_hardware_interface('bios')

    def test_baremetal_reset_boot_interface(self):
        self._test_baremetal_reset_hardware_interface('boot')

    def test_baremetal_reset_console_interface(self):
        self._test_baremetal_reset_hardware_interface('console')

    def test_baremetal_reset_deploy_interface(self):
        self._test_baremetal_reset_hardware_interface('deploy')

    def test_baremetal_reset_firmware_interface(self):
        self._test_baremetal_reset_hardware_interface('firmware')

    def test_baremetal_reset_inspect_interface(self):
        self._test_baremetal_reset_hardware_interface('inspect')

    def test_baremetal_reset_management_interface(self):
        self._test_baremetal_reset_hardware_interface('management')

    def test_baremetal_reset_network_interface(self):
        self._test_baremetal_reset_hardware_interface('network')

    def test_baremetal_reset_power_interface(self):
        self._test_baremetal_reset_hardware_interface('power')

    def test_baremetal_reset_raid_interface(self):
        self._test_baremetal_reset_hardware_interface('raid')

    def test_baremetal_reset_rescue_interface(self):
        self._test_baremetal_reset_hardware_interface('rescue')

    def test_baremetal_reset_storage_interface(self):
        self._test_baremetal_reset_hardware_interface('storage')

    def test_baremetal_reset_vendor_interface(self):
        self._test_baremetal_reset_hardware_interface('vendor')

    def test_baremetal_set_resource_class(self):
        arglist = ['node_uuid', '--resource-class', 'foo']
        verifylist = [('nodes', ['node_uuid']), ('resource_class', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/resource_class', 'value': 'foo', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_conductor_group(self):
        arglist = ['node_uuid', '--conductor-group', 'foo']
        verifylist = [('nodes', ['node_uuid']), ('conductor_group', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/conductor_group', 'value': 'foo', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_automated_clean(self):
        arglist = ['node_uuid', '--automated-clean']
        verifylist = [('nodes', ['node_uuid']), ('automated_clean', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/automated_clean', 'value': 'True', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_no_automated_clean(self):
        arglist = ['node_uuid', '--no-automated-clean']
        verifylist = [('nodes', ['node_uuid']), ('automated_clean', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/automated_clean', 'value': 'False', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_protected(self):
        arglist = ['node_uuid', '--protected']
        verifylist = [('nodes', ['node_uuid']), ('protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/protected', 'value': 'True', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_protected_with_reason(self):
        arglist = ['node_uuid', '--protected', '--protected-reason', 'reason!']
        verifylist = [('nodes', ['node_uuid']), ('protected', True), ('protected_reason', 'reason!')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/protected', 'value': 'True', 'op': 'add'}, {'path': '/protected_reason', 'value': 'reason!', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_retired(self):
        arglist = ['node_uuid', '--retired']
        verifylist = [('nodes', ['node_uuid']), ('retired', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/retired', 'value': 'True', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_retired_with_reason(self):
        arglist = ['node_uuid', '--retired', '--retired-reason', 'out of warranty!']
        verifylist = [('nodes', ['node_uuid']), ('retired', True), ('retired_reason', 'out of warranty!')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/retired', 'value': 'True', 'op': 'add'}, {'path': '/retired_reason', 'value': 'out of warranty!', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_extra(self):
        arglist = ['node_uuid', '--extra', 'foo=bar']
        verifylist = [('nodes', ['node_uuid']), ('extra', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_driver_info(self):
        arglist = ['node_uuid', '--driver-info', 'foo=bar']
        verifylist = [('nodes', ['node_uuid']), ('driver_info', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/driver_info/foo', 'value': 'bar', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_instance_info(self):
        arglist = ['node_uuid', '--instance-info', 'foo=bar']
        verifylist = [('nodes', ['node_uuid']), ('instance_info', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/instance_info/foo', 'value': 'bar', 'op': 'add'}], reset_interfaces=None)

    @mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
    @mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
    def test_baremetal_set_target_raid_config(self, mock_handle, mock_stdin):
        self.cmd.log = mock.Mock(autospec=True)
        target_raid_config_string = '{"raid": "config"}'
        expected_target_raid_config = {'raid': 'config'}
        mock_handle.return_value = expected_target_raid_config.copy()
        arglist = ['node_uuid', '--target-raid-config', target_raid_config_string]
        verifylist = [('nodes', ['node_uuid']), ('target_raid_config', target_raid_config_string)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cmd.log.warning.assert_not_called()
        self.assertFalse(mock_stdin.called)
        mock_handle.assert_called_once_with(target_raid_config_string)
        self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', expected_target_raid_config)
        self.assertFalse(self.baremetal_mock.node.update.called)

    @mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
    @mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
    def test_baremetal_set_target_raid_config_and_name(self, mock_handle, mock_stdin):
        self.cmd.log = mock.Mock(autospec=True)
        target_raid_config_string = '{"raid": "config"}'
        expected_target_raid_config = {'raid': 'config'}
        mock_handle.return_value = expected_target_raid_config.copy()
        arglist = ['node_uuid', '--name', 'xxxxx', '--target-raid-config', target_raid_config_string]
        verifylist = [('nodes', ['node_uuid']), ('name', 'xxxxx'), ('target_raid_config', target_raid_config_string)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cmd.log.warning.assert_not_called()
        self.assertFalse(mock_stdin.called)
        mock_handle.assert_called_once_with(target_raid_config_string)
        self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', expected_target_raid_config)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/name', 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=None)

    @mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
    @mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
    def test_baremetal_set_target_raid_config_stdin(self, mock_handle, mock_stdin):
        self.cmd.log = mock.Mock(autospec=True)
        target_value = '-'
        target_raid_config_string = '{"raid": "config"}'
        expected_target_raid_config = {'raid': 'config'}
        mock_stdin.return_value = target_raid_config_string
        mock_handle.return_value = expected_target_raid_config.copy()
        arglist = ['node_uuid', '--target-raid-config', target_value]
        verifylist = [('nodes', ['node_uuid']), ('target_raid_config', target_value)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cmd.log.warning.assert_not_called()
        mock_stdin.assert_called_once_with('target_raid_config')
        mock_handle.assert_called_once_with(target_raid_config_string)
        self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', expected_target_raid_config)
        self.assertFalse(self.baremetal_mock.node.update.called)

    @mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
    @mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
    def test_baremetal_set_target_raid_config_stdin_exception(self, mock_handle, mock_stdin):
        self.cmd.log = mock.Mock(autospec=True)
        target_value = '-'
        mock_stdin.side_effect = exc.InvalidAttribute('bad')
        arglist = ['node_uuid', '--target-raid-config', target_value]
        verifylist = [('nodes', ['node_uuid']), ('target_raid_config', target_value)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.InvalidAttribute, self.cmd.take_action, parsed_args)
        self.cmd.log.warning.assert_not_called()
        mock_stdin.assert_called_once_with('target_raid_config')
        self.assertFalse(mock_handle.called)
        self.assertFalse(self.baremetal_mock.node.set_target_raid_config.called)
        self.assertFalse(self.baremetal_mock.node.update.called)

    def test_baremetal_set_owner(self):
        arglist = ['node_uuid', '--owner', 'owner 1']
        verifylist = [('nodes', ['node_uuid']), ('owner', 'owner 1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/owner', 'value': 'owner 1', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_description(self):
        arglist = ['node_uuid', '--description', 'there is no spoon']
        verifylist = [('nodes', ['node_uuid']), ('description', 'there is no spoon')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/description', 'value': 'there is no spoon', 'op': 'add'}], reset_interfaces=None)

    def test_baremetal_set_lessee(self):
        arglist = ['node_uuid', '--lessee', 'lessee 1']
        verifylist = [('nodes', ['node_uuid']), ('lessee', 'lessee 1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'op': 'add', 'path': '/lessee', 'value': 'lessee 1'}], reset_interfaces=None)

    def test_baremetal_set_shard(self):
        arglist = ['node_uuid', '--shard', 'myshard']
        verifylist = [('nodes', ['node_uuid']), ('shard', 'myshard')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'op': 'add', 'path': '/shard', 'value': 'myshard'}], reset_interfaces=None)

    @mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
    @mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
    def test_baremetal_set_network_data(self, mock_handle, mock_stdin):
        self.cmd.log = mock.Mock(autospec=True)
        network_data_string = '{"a": ["b"]}'
        expected_network_data = {'a': ['b']}
        mock_handle.return_value = expected_network_data.copy()
        arglist = ['node_uuid', '--network-data', network_data_string]
        verifylist = [('nodes', ['node_uuid']), ('network_data', network_data_string)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/network_data', 'value': expected_network_data, 'op': 'add'}], reset_interfaces=None)