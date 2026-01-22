import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteNetwork(TestNetwork):

    def setUp(self):
        super(TestDeleteNetwork, self).setUp()
        self._networks = network_fakes.create_networks(count=3)
        self.network_client.delete_network = mock.Mock(return_value=None)
        self.network_client.find_network = network_fakes.get_networks(networks=self._networks)
        self.cmd = network.DeleteNetwork(self.app, self.namespace)

    def test_delete_one_network(self):
        arglist = [self._networks[0].name]
        verifylist = [('network', [self._networks[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_network.assert_called_once_with(self._networks[0])
        self.assertIsNone(result)

    def test_delete_multiple_networks(self):
        arglist = []
        for n in self._networks:
            arglist.append(n.id)
        verifylist = [('network', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for n in self._networks:
            calls.append(call(n))
        self.network_client.delete_network.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_networks_exception(self):
        arglist = [self._networks[0].id, 'xxxx-yyyy-zzzz', self._networks[1].id]
        verifylist = [('network', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ret_find = [self._networks[0], exceptions.NotFound('404'), self._networks[1]]
        self.network_client.find_network = mock.Mock(side_effect=ret_find)
        ret_delete = [None, exceptions.NotFound('404')]
        self.network_client.delete_network = mock.Mock(side_effect=ret_delete)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        calls = [call(self._networks[0]), call(self._networks[1])]
        self.network_client.delete_network.assert_has_calls(calls)