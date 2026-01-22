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
class TestUnsetPort(TestPort):

    def setUp(self):
        super(TestUnsetPort, self).setUp()
        self._testport = network_fakes.create_one_port({'fixed_ips': [{'subnet_id': '042eb10a-3a18-4658-ab-cf47c8d03152', 'ip_address': '0.0.0.1'}, {'subnet_id': '042eb10a-3a18-4658-ab-cf47c8d03152', 'ip_address': '1.0.0.0'}], 'binding:profile': {'batman': 'Joker', 'Superman': 'LexLuthor'}, 'tags': ['green', 'red']})
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet({'id': '042eb10a-3a18-4658-ab-cf47c8d03152'})
        self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
        self.network_client.find_port = mock.Mock(return_value=self._testport)
        self.network_client.update_port = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = port.UnsetPort(self.app, self.namespace)

    def test_unset_port_parameters(self):
        arglist = ['--fixed-ip', 'subnet=042eb10a-3a18-4658-ab-cf47c8d03152,ip-address=1.0.0.0', '--binding-profile', 'Superman', '--qos-policy', '--host', self._testport.name]
        verifylist = [('fixed_ip', [{'subnet': '042eb10a-3a18-4658-ab-cf47c8d03152', 'ip-address': '1.0.0.0'}]), ('binding_profile', ['Superman']), ('qos_policy', True), ('host', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'fixed_ips': [{'subnet_id': '042eb10a-3a18-4658-ab-cf47c8d03152', 'ip_address': '0.0.0.1'}], 'binding:profile': {'batman': 'Joker'}, 'qos_policy_id': None, 'binding:host_id': None}
        self.network_client.update_port.assert_called_once_with(self._testport, **attrs)
        self.assertIsNone(result)

    def test_unset_port_fixed_ip_not_existent(self):
        arglist = ['--fixed-ip', 'ip-address=1.0.0.1', '--binding-profile', 'Superman', self._testport.name]
        verifylist = [('fixed_ip', [{'ip-address': '1.0.0.1'}]), ('binding_profile', ['Superman'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_port_binding_profile_not_existent(self):
        arglist = ['--fixed-ip', 'ip-address=1.0.0.0', '--binding-profile', 'Neo', self._testport.name]
        verifylist = [('fixed_ip', [{'ip-address': '1.0.0.0'}]), ('binding_profile', ['Neo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_security_group(self):
        _fake_sg1 = network_fakes.FakeSecurityGroup.create_one_security_group()
        _fake_sg2 = network_fakes.FakeSecurityGroup.create_one_security_group()
        _fake_port = network_fakes.create_one_port({'security_group_ids': [_fake_sg1.id, _fake_sg2.id]})
        self.network_client.find_port = mock.Mock(return_value=_fake_port)
        self.network_client.find_security_group = mock.Mock(return_value=_fake_sg2)
        arglist = ['--security-group', _fake_sg2.id, _fake_port.name]
        verifylist = [('security_group_ids', [_fake_sg2.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'security_group_ids': [_fake_sg1.id]}
        self.network_client.update_port.assert_called_once_with(_fake_port, **attrs)
        self.assertIsNone(result)

    def test_unset_port_security_group_not_existent(self):
        _fake_sg1 = network_fakes.FakeSecurityGroup.create_one_security_group()
        _fake_sg2 = network_fakes.FakeSecurityGroup.create_one_security_group()
        _fake_port = network_fakes.create_one_port({'security_group_ids': [_fake_sg1.id]})
        self.network_client.find_security_group = mock.Mock(return_value=_fake_sg2)
        arglist = ['--security-group', _fake_sg2.id, _fake_port.name]
        verifylist = [('security_group_ids', [_fake_sg2.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_port_allowed_address_pair(self):
        _fake_port = network_fakes.create_one_port({'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]})
        self.network_client.find_port = mock.Mock(return_value=_fake_port)
        arglist = ['--allowed-address', 'ip-address=192.168.1.123', _fake_port.name]
        verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.123'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'allowed_address_pairs': []}
        self.network_client.update_port.assert_called_once_with(_fake_port, **attrs)
        self.assertIsNone(result)

    def test_unset_port_allowed_address_pair_not_existent(self):
        _fake_port = network_fakes.create_one_port({'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]})
        self.network_client.find_port = mock.Mock(return_value=_fake_port)
        arglist = ['--allowed-address', 'ip-address=192.168.1.45', _fake_port.name]
        verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.45'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_port_data_plane_status(self):
        _fake_port = network_fakes.create_one_port({'data_plane_status': 'ACTIVE'})
        self.network_client.find_port = mock.Mock(return_value=_fake_port)
        arglist = ['--data-plane-status', _fake_port.name]
        verifylist = [('data_plane_status', True), ('port', _fake_port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'data_plane_status': None}
        self.network_client.update_port.assert_called_once_with(_fake_port, **attrs)
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
        arglist.append(self._testport.name)
        verifylist.append(('port', self._testport.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_port.called)
        self.network_client.set_tags.assert_called_once_with(self._testport, test_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_unset_with_tags(self):
        self._test_unset_tags(with_tags=True)

    def test_unset_with_all_tag(self):
        self._test_unset_tags(with_tags=False)

    def test_unset_numa_affinity_policy(self):
        _fake_port = network_fakes.create_one_port({'numa_affinity_policy': 'required'})
        self.network_client.find_port = mock.Mock(return_value=_fake_port)
        arglist = ['--numa-policy', _fake_port.name]
        verifylist = [('numa_policy', True), ('port', _fake_port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'numa_affinity_policy': None}
        self.network_client.update_port.assert_called_once_with(_fake_port, **attrs)
        self.assertIsNone(result)

    def test_unset_hints(self):
        testport = network_fakes.create_one_port()
        self.network_client.find_port = mock.Mock(return_value=testport)
        arglist = ['--hints', testport.name]
        verifylist = [('hints', True), ('port', testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_port.assert_called_once_with(testport, **{'hints': None})
        self.assertIsNone(result)