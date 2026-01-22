import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortGroupList(TestBaremetalPortGroup):

    def setUp(self):
        super(TestBaremetalPortGroupList, self).setUp()
        self.baremetal_mock.portgroup.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.PORTGROUP), loaded=True)]
        self.cmd = baremetal_portgroup.ListBaremetalPortGroup(self.app, None)

    def test_baremetal_portgroup_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Address', 'Name')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_address, baremetal_fakes.baremetal_portgroup_name),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_portgroup_list_address(self):
        arglist = ['--address', baremetal_fakes.baremetal_portgroup_address]
        verifylist = [('address', baremetal_fakes.baremetal_portgroup_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'address': baremetal_fakes.baremetal_portgroup_address, 'marker': None, 'limit': None}
        self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Address', 'Name')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_address, baremetal_fakes.baremetal_portgroup_name),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_portgroup_list_node(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'node': baremetal_fakes.baremetal_uuid, 'marker': None, 'limit': None}
        self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Address', 'Name')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_address, baremetal_fakes.baremetal_portgroup_name),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_portgroup_list_long(self):
        arglist = ['--long']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Address', 'Created At', 'Extra', 'Standalone Ports Supported', 'Node UUID', 'Name', 'Updated At', 'Internal Info', 'Mode', 'Properties')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_address, '', baremetal_fakes.baremetal_portgroup_extra, '', baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_portgroup_name, '', '', baremetal_fakes.baremetal_portgroup_mode, baremetal_fakes.baremetal_portgroup_properties),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_portgroup_list_fields(self):
        arglist = ['--fields', 'uuid', 'address']
        verifylist = [('fields', [['uuid', 'address']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'address')}
        self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)

    def test_baremetal_portgroup_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'address', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'address'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'address', 'extra')}
        self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)

    def test_baremetal_portgroup_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)