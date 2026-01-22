from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestUnsetFloatingIP(TestFloatingIPNetwork):
    floating_network = network_fakes.create_one_network()
    subnet = network_fakes.FakeSubnet.create_one_subnet()
    port = network_fakes.create_one_port()
    floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip(attrs={'floating_network_id': floating_network.id, 'port_id': port.id, 'tags': ['green', 'red']})

    def setUp(self):
        super(TestUnsetFloatingIP, self).setUp()
        self.network_client.find_ip = mock.Mock(return_value=self.floating_ip)
        self.network_client.update_ip = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = fip.UnsetFloatingIP(self.app, self.namespace)

    def test_floating_ip_unset_port(self):
        arglist = [self.floating_ip.id, '--port']
        verifylist = [('floating_ip', self.floating_ip.id), ('port', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'port_id': None}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)
        self.assertIsNone(result)

    def test_floating_ip_unset_qos_policy(self):
        arglist = [self.floating_ip.id, '--qos-policy']
        verifylist = [('floating_ip', self.floating_ip.id), ('qos_policy', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'qos_policy_id': None}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)
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
        arglist.append(self.floating_ip.id)
        verifylist.append(('floating_ip', self.floating_ip.id))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_ip.called)
        self.network_client.set_tags.assert_called_once_with(self.floating_ip, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_unset_with_tags(self):
        self._test_unset_tags(with_tags=True)

    def test_unset_with_all_tag(self):
        self._test_unset_tags(with_tags=False)