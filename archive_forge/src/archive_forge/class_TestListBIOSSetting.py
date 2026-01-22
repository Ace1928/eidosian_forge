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
class TestListBIOSSetting(TestBaremetal):

    def setUp(self):
        super(TestListBIOSSetting, self).setUp()
        self.baremetal_mock.node.list_bios_settings.return_value = baremetal_fakes.BIOS_SETTINGS
        self.cmd = baremetal_node.ListBIOSSettingBaremetalNode(self.app, None)

    def test_baremetal_list_bios_setting(self):
        arglist = ['node_uuid']
        verifylist = [('node', 'node_uuid')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.list_bios_settings.assert_called_once_with('node_uuid')
        expected_columns = ('BIOS setting name', 'BIOS setting value')
        self.assertEqual(expected_columns, columns)
        expected_data = [(s['name'], s['value']) for s in baremetal_fakes.BIOS_SETTINGS]
        self.assertEqual(tuple(expected_data), tuple(data))

    def test_baremetal_list_bios_setting_long(self):
        verifylist = [('long', True)]
        arglist = ['node_uuid', '--long']
        verifylist = [('node', 'node_uuid'), ('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.baremetal_mock.node.list_bios_settings.return_value = baremetal_fakes.BIOS_DETAILED_SETTINGS
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True}
        self.baremetal_mock.node.list_bios_settings.assert_called_once_with('node_uuid', **kwargs)
        expected_columns = ('Name', 'Value', 'Attribute Type', 'Allowable Values', 'Lower Bound', 'Minimum Length', 'Maximum Length', 'Read Only', 'Reset Required', 'Unique', 'Upper Bound')
        self.assertEqual(expected_columns, columns)
        expected_data = (('SysName', 'my-system', 'String', '', '', '1', '16', '', '', '', ''), ('NumCores', '10', 'Integer', '', '10', '', '', '', '', '', '20'), ('ProcVirtualization', 'Enabled', 'Enumeration', ['Enabled', 'Disabled'], '', '', '', '', '', '', ''))
        self.assertEqual(expected_data, tuple(data))

    def test_baremetal_list_bios_setting_fields(self):
        arglist = ['node_uuid', '--fields', 'name', 'attribute_type']
        verifylist = [('fields', [['name', 'attribute_type']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.baremetal_mock.node.list_bios_settings.return_value = baremetal_fakes.BIOS_DETAILED_SETTINGS
        columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('Value', columns)
        self.assertIn('Name', columns)
        self.assertIn('Attribute Type', columns)
        kwargs = {'detail': False, 'fields': ('name', 'attribute_type')}
        self.baremetal_mock.node.list_bios_settings.assert_called_with('node_uuid', **kwargs)