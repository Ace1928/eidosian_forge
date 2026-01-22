from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestDeletePort(TestPort):
    _ports = network_fakes.create_ports(count=2)

    def setUp(self):
        super(TestDeletePort, self).setUp()
        self.network_client.delete_port = mock.Mock(return_value=None)
        self.network_client.find_port = network_fakes.get_ports(ports=self._ports)
        self.cmd = port.DeletePort(self.app, self.namespace)

    def test_port_delete(self):
        arglist = [self._ports[0].name]
        verifylist = [('port', [self._ports[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_port.assert_called_once_with(self._ports[0].name, ignore_missing=False)
        self.network_client.delete_port.assert_called_once_with(self._ports[0])
        self.assertIsNone(result)

    def test_multi_ports_delete(self):
        arglist = []
        verifylist = []
        for p in self._ports:
            arglist.append(p.name)
        verifylist = [('port', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for p in self._ports:
            calls.append(call(p))
        self.network_client.delete_port.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_ports_delete_with_exception(self):
        arglist = [self._ports[0].name, 'unexist_port']
        verifylist = [('port', [self._ports[0].name, 'unexist_port'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._ports[0], exceptions.CommandError]
        self.network_client.find_port = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 ports failed to delete.', str(e))
        self.network_client.find_port.assert_any_call(self._ports[0].name, ignore_missing=False)
        self.network_client.find_port.assert_any_call('unexist_port', ignore_missing=False)
        self.network_client.delete_port.assert_called_once_with(self._ports[0])