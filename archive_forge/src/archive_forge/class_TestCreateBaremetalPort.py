import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestCreateBaremetalPort(TestBaremetalPort):

    def setUp(self):
        super(TestCreateBaremetalPort, self).setUp()
        self.baremetal_mock.port.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_PORT), loaded=True)
        self.cmd = baremetal_port.CreateBaremetalPort(self.app, None)

    def test_baremetal_port_create(self):
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid}
        self.baremetal_mock.port.create.assert_called_once_with(**args)

    def test_baremetal_port_create_extras(self):
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--extra', 'key1=value1', '--extra', 'key2=value2']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('extra', ['key1=value1', 'key2=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'extra': baremetal_fakes.baremetal_port_extra}
        self.baremetal_mock.port.create.assert_called_once_with(**args)

    def test_baremetal_port_create_no_address(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid)]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_port_create_no_node(self):
        arglist = [baremetal_fakes.baremetal_port_address]
        verifylist = [('address', baremetal_fakes.baremetal_port_address)]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_port_create_no_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_port_create_uuid(self):
        port_uuid = 'da6c8d2e-fbcd-457a-b2a7-cc5c775933af'
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--uuid', port_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('uuid', port_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'uuid': port_uuid}
        self.baremetal_mock.port.create.assert_called_once_with(**args)

    def _test_baremetal_port_create_llc_warning(self, additional_args, additional_verify_items):
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid]
        arglist.extend(additional_args)
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address)]
        verifylist.extend(additional_verify_items)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.log = mock.Mock()
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'local_link_connection': {'switch_id': 'aa:bb:cc:dd:ee:ff', 'port_id': 'eth0'}}
        self.baremetal_mock.port.create.assert_called_once_with(**args)
        self.cmd.log.warning.assert_called()

    def test_baremetal_port_create_llc_warning_some_deprecated(self):
        self._test_baremetal_port_create_llc_warning(additional_args=['-l', 'port_id=eth0', '--local-link-connection', 'switch_id=aa:bb:cc:dd:ee:ff'], additional_verify_items=[('local_link_connection_deprecated', ['port_id=eth0']), ('local_link_connection', ['switch_id=aa:bb:cc:dd:ee:ff'])])

    def test_baremetal_port_create_llc_warning_all_deprecated(self):
        self._test_baremetal_port_create_llc_warning(additional_args=['-l', 'port_id=eth0', '-l', 'switch_id=aa:bb:cc:dd:ee:ff'], additional_verify_items=[('local_link_connection_deprecated', ['port_id=eth0', 'switch_id=aa:bb:cc:dd:ee:ff'])])

    def test_baremetal_port_create_portgroup_uuid(self):
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--port-group', baremetal_fakes.baremetal_portgroup_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('portgroup_uuid', baremetal_fakes.baremetal_portgroup_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'portgroup_uuid': baremetal_fakes.baremetal_portgroup_uuid}
        self.baremetal_mock.port.create.assert_called_once_with(**args)

    def test_baremetal_port_create_physical_network(self):
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--physical-network', baremetal_fakes.baremetal_port_physical_network]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('physical_network', baremetal_fakes.baremetal_port_physical_network)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'physical_network': baremetal_fakes.baremetal_port_physical_network}
        self.baremetal_mock.port.create.assert_called_once_with(**args)

    def test_baremetal_port_create_smartnic(self):
        arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--is-smartnic']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('is_smartnic', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'is_smartnic': True}
        self.baremetal_mock.port.create.assert_called_once_with(**args)