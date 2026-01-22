import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestListBaremetalDriver(TestBaremetalDriver):

    def setUp(self):
        super(TestListBaremetalDriver, self).setUp()
        self.baremetal_mock.driver.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_DRIVER), loaded=True)]
        self.cmd = baremetal_driver.ListBaremetalDriver(self.app, None)

    def test_baremetal_driver_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('Supported driver(s)', 'Active host(s)')
        self.assertEqual(collist, tuple(columns))
        datalist = ((baremetal_fakes.baremetal_driver_name, ', '.join(baremetal_fakes.baremetal_driver_hosts)),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_driver_list_with_type(self):
        arglist = ['--type', baremetal_fakes.baremetal_driver_type]
        verifylist = [('type', baremetal_fakes.baremetal_driver_type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('Supported driver(s)', 'Active host(s)')
        self.assertEqual(collist, tuple(columns))
        datalist = ((baremetal_fakes.baremetal_driver_name, ', '.join(baremetal_fakes.baremetal_driver_hosts)),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_driver_list_with_detail(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('Supported driver(s)', 'Type', 'Active host(s)', 'Default BIOS Interface', 'Default Boot Interface', 'Default Console Interface', 'Default Deploy Interface', 'Default Firmware Interface', 'Default Inspect Interface', 'Default Management Interface', 'Default Network Interface', 'Default Power Interface', 'Default RAID Interface', 'Default Rescue Interface', 'Default Storage Interface', 'Default Vendor Interface', 'Enabled BIOS Interfaces', 'Enabled Boot Interfaces', 'Enabled Console Interfaces', 'Enabled Deploy Interfaces', 'Enabled Firmware Interfaces', 'Enabled Inspect Interfaces', 'Enabled Management Interfaces', 'Enabled Network Interfaces', 'Enabled Power Interfaces', 'Enabled RAID Interfaces', 'Enabled Rescue Interfaces', 'Enabled Storage Interfaces', 'Enabled Vendor Interfaces')
        self.assertEqual(collist, tuple(columns))
        datalist = ((baremetal_fakes.baremetal_driver_name, baremetal_fakes.baremetal_driver_type, ', '.join(baremetal_fakes.baremetal_driver_hosts), baremetal_fakes.baremetal_driver_default_bios_if, baremetal_fakes.baremetal_driver_default_boot_if, baremetal_fakes.baremetal_driver_default_console_if, baremetal_fakes.baremetal_driver_default_deploy_if, baremetal_fakes.baremetal_driver_default_firmware_if, baremetal_fakes.baremetal_driver_default_inspect_if, baremetal_fakes.baremetal_driver_default_management_if, baremetal_fakes.baremetal_driver_default_network_if, baremetal_fakes.baremetal_driver_default_power_if, baremetal_fakes.baremetal_driver_default_raid_if, baremetal_fakes.baremetal_driver_default_rescue_if, baremetal_fakes.baremetal_driver_default_storage_if, baremetal_fakes.baremetal_driver_default_vendor_if, ', '.join(baremetal_fakes.baremetal_driver_enabled_bios_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_boot_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_console_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_deploy_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_firmware_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_inspect_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_management_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_network_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_power_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_raid_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_rescue_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_storage_ifs), ', '.join(baremetal_fakes.baremetal_driver_enabled_vendor_ifs)),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_driver_list_fields(self):
        arglist = ['--fields', 'name', 'hosts']
        verifylist = [('fields', [['name', 'hosts']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'driver_type': None, 'detail': None, 'fields': ('name', 'hosts')}
        self.baremetal_mock.driver.list.assert_called_with(**kwargs)

    def test_baremetal_driver_list_fields_multiple(self):
        arglist = ['--fields', 'name', '--fields', 'hosts', 'type']
        verifylist = [('fields', [['name'], ['hosts', 'type']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'driver_type': None, 'detail': None, 'fields': ('name', 'hosts', 'type')}
        self.baremetal_mock.driver.list.assert_called_with(**kwargs)

    def test_baremetal_driver_list_invalid_fields(self):
        arglist = ['--fields', 'name', 'invalid']
        verifylist = [('fields', [['name', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_driver_list_fields_with_long(self):
        arglist = ['--fields', 'name', 'hosts', '--long']
        verifylist = [('fields', [['name', 'invalid']]), ('long', True)]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)