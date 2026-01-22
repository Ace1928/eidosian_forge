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
class TestUnsetNetwork(TestNetwork):
    _network = network_fakes.create_one_network({'tags': ['green', 'red']})
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy(attrs={'id': _network.qos_policy_id})

    def setUp(self):
        super(TestUnsetNetwork, self).setUp()
        self.network_client.update_network = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.network_client.find_network = mock.Mock(return_value=self._network)
        self.network_client.find_qos_policy = mock.Mock(return_value=self.qos_policy)
        self.cmd = network.UnsetNetwork(self.app, self.namespace)

    def test_unset_nothing(self):
        arglist = [self._network.name]
        verifylist = [('network', self._network.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_network.called)
        self.assertFalse(self.network_client.set_tags.called)
        self.assertIsNone(result)

    def _test_unset_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['green']
        else:
            arglist = ['--all-tag']
            verifylist = [('all_tag', True)]
            expected_args = []
        arglist.append(self._network.name)
        verifylist.append(('network', self._network.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_network.called)
        self.network_client.set_tags.assert_called_once_with(self._network, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_unset_with_tags(self):
        self._test_unset_tags(with_tags=True)

    def test_unset_with_all_tag(self):
        self._test_unset_tags(with_tags=False)