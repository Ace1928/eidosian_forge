import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortSet(TestBaremetalPort):

    def setUp(self):
        super(TestBaremetalPortSet, self).setUp()
        self.baremetal_mock.port.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_PORT), loaded=True)
        self.cmd = baremetal_port.SetBaremetalPort(self.app, None)

    def test_baremetal_port_set_node_uuid(self):
        new_node_uuid = '1111-111111-1111'
        arglist = [baremetal_fakes.baremetal_port_uuid, '--node', new_node_uuid]
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('node_uuid', new_node_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/node_uuid', 'value': new_node_uuid, 'op': 'add'}])

    def test_baremetal_port_set_address(self):
        arglist = [baremetal_fakes.baremetal_port_uuid, '--address', baremetal_fakes.baremetal_port_address]
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('address', baremetal_fakes.baremetal_port_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/address', 'value': baremetal_fakes.baremetal_port_address, 'op': 'add'}])

    def test_baremetal_set_extra(self):
        arglist = ['port', '--extra', 'foo=bar']
        verifylist = [('port', 'port'), ('extra', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])

    def test_baremetal_port_set_portgroup_uuid(self):
        new_portgroup_uuid = '1111-111111-1111'
        arglist = [baremetal_fakes.baremetal_port_uuid, '--port-group', new_portgroup_uuid]
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('portgroup_uuid', new_portgroup_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/portgroup_uuid', 'value': new_portgroup_uuid, 'op': 'add'}])

    def test_baremetal_set_local_link_connection(self):
        arglist = [baremetal_fakes.baremetal_port_uuid, '--local-link-connection', 'switch_info=bar']
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('local_link_connection', ['switch_info=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/local_link_connection/switch_info', 'value': 'bar', 'op': 'add'}])

    def test_baremetal_port_set_pxe_enabled(self):
        arglist = [baremetal_fakes.baremetal_port_uuid, '--pxe-enabled']
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('pxe_enabled', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/pxe_enabled', 'value': 'True', 'op': 'add'}])

    def test_baremetal_port_set_pxe_disabled(self):
        arglist = [baremetal_fakes.baremetal_port_uuid, '--pxe-disabled']
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('pxe_enabled', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/pxe_enabled', 'value': 'False', 'op': 'add'}])

    def test_baremetal_port_set_physical_network(self):
        new_physical_network = 'physnet2'
        arglist = [baremetal_fakes.baremetal_port_uuid, '--physical-network', new_physical_network]
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('physical_network', new_physical_network)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/physical_network', 'value': new_physical_network, 'op': 'add'}])

    def test_baremetal_port_set_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_port_set_no_property(self):
        arglist = [baremetal_fakes.baremetal_port_uuid]
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.port.update.called)

    def test_baremetal_port_set_is_smartnic(self):
        arglist = [baremetal_fakes.baremetal_port_uuid, '--is-smartnic']
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('is_smartnic', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/is_smartnic', 'value': 'True', 'op': 'add'}])