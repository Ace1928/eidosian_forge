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
def test_set_this(self):
    arglist = [self._network.name, '--enable', '--name', 'noob', '--share', '--description', self._network.description, '--dns-domain', 'example.org.', '--external', '--default', '--provider-network-type', 'vlan', '--provider-physical-network', 'physnet1', '--provider-segment', '400', '--enable-port-security', '--qos-policy', self.qos_policy.name]
    verifylist = [('network', self._network.name), ('enable', True), ('description', self._network.description), ('name', 'noob'), ('share', True), ('external', True), ('default', True), ('provider_network_type', 'vlan'), ('physical_network', 'physnet1'), ('segmentation_id', '400'), ('enable_port_security', True), ('qos_policy', self.qos_policy.name), ('dns_domain', 'example.org.')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'noob', 'admin_state_up': True, 'description': self._network.description, 'shared': True, 'router:external': True, 'is_default': True, 'provider:network_type': 'vlan', 'provider:physical_network': 'physnet1', 'provider:segmentation_id': '400', 'port_security_enabled': True, 'qos_policy_id': self.qos_policy.id, 'dns_domain': 'example.org.'}
    self.network_client.update_network.assert_called_once_with(self._network, **attrs)
    self.assertIsNone(result)