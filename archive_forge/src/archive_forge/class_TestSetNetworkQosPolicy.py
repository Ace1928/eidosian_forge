from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkQosPolicy(TestQosPolicy):
    _qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()

    def setUp(self):
        super(TestSetNetworkQosPolicy, self).setUp()
        self.network_client.update_qos_policy = mock.Mock(return_value=None)
        self.network_client.find_qos_policy = mock.Mock(return_value=self._qos_policy)
        self.cmd = network_qos_policy.SetNetworkQosPolicy(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self._qos_policy.name]
        verifylist = [('policy', self._qos_policy.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {}
        self.network_client.update_qos_policy.assert_called_with(self._qos_policy, **attrs)
        self.assertIsNone(result)

    def test_set_name_share_description_default(self):
        arglist = ['--name', 'new_qos_policy', '--share', '--description', 'QoS policy description', '--default', self._qos_policy.name]
        verifylist = [('name', 'new_qos_policy'), ('share', True), ('description', 'QoS policy description'), ('default', True), ('policy', self._qos_policy.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'new_qos_policy', 'description': 'QoS policy description', 'shared': True, 'is_default': True}
        self.network_client.update_qos_policy.assert_called_with(self._qos_policy, **attrs)
        self.assertIsNone(result)

    def test_set_no_share_no_default(self):
        arglist = ['--no-share', '--no-default', self._qos_policy.name]
        verifylist = [('no_share', True), ('no_default', True), ('policy', self._qos_policy.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'shared': False, 'is_default': False}
        self.network_client.update_qos_policy.assert_called_with(self._qos_policy, **attrs)
        self.assertIsNone(result)