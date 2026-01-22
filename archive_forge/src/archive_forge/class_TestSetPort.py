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
class TestSetPort(TestPort):
    _port = network_fakes.create_one_port({'tags': ['green', 'red']})

    def setUp(self):
        super(TestSetPort, self).setUp()
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet()
        self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
        self.network_client.find_port = mock.Mock(return_value=self._port)
        self.network_client.update_port = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = port.SetPort(self.app, self.namespace)

    def test_set_port_defaults(self):
        arglist = [self._port.name]
        verifylist = [('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_port.called)
        self.assertFalse(self.network_client.set_tags.called)
        self.assertIsNone(result)

    def test_set_port_fixed_ip(self):
        _testport = network_fakes.create_one_port({'fixed_ips': [{'ip_address': '0.0.0.1'}]})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--fixed-ip', 'ip-address=10.0.0.12', _testport.name]
        verifylist = [('fixed_ip', [{'ip-address': '10.0.0.12'}]), ('port', _testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'fixed_ips': [{'ip_address': '0.0.0.1'}, {'ip_address': '10.0.0.12'}]}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_fixed_ip_clear(self):
        _testport = network_fakes.create_one_port({'fixed_ips': [{'ip_address': '0.0.0.1'}]})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--fixed-ip', 'ip-address=10.0.0.12', '--no-fixed-ip', _testport.name]
        verifylist = [('fixed_ip', [{'ip-address': '10.0.0.12'}]), ('no_fixed_ip', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'fixed_ips': [{'ip_address': '10.0.0.12'}]}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_dns_name(self):
        arglist = ['--dns-name', '8.8.8.8', self._port.name]
        verifylist = [('dns_name', '8.8.8.8'), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'dns_name': '8.8.8.8'}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_overwrite_binding_profile(self):
        _testport = network_fakes.create_one_port({'binding_profile': {'lok_i': 'visi_on'}})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--binding-profile', 'lok_i=than_os', '--no-binding-profile', _testport.name]
        verifylist = [('binding_profile', {'lok_i': 'than_os'}), ('no_binding_profile', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'binding:profile': {'lok_i': 'than_os'}}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_overwrite_mac_address(self):
        _testport = network_fakes.create_one_port({'mac_address': '11:22:33:44:55:66'})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--mac-address', '66:55:44:33:22:11', _testport.name]
        verifylist = [('mac_address', '66:55:44:33:22:11')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'mac_address': '66:55:44:33:22:11'}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_this(self):
        arglist = ['--disable', '--no-fixed-ip', '--no-binding-profile', self._port.name]
        verifylist = [('disable', True), ('no_binding_profile', True), ('no_fixed_ip', True), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': False, 'binding:profile': {}, 'fixed_ips': []}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_that(self):
        arglist = ['--description', 'newDescription', '--enable', '--vnic-type', 'macvtap', '--binding-profile', 'foo=bar', '--host', 'binding-host-id-xxxx', '--name', 'newName', self._port.name]
        verifylist = [('description', 'newDescription'), ('enable', True), ('vnic_type', 'macvtap'), ('binding_profile', {'foo': 'bar'}), ('host', 'binding-host-id-xxxx'), ('name', 'newName'), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': True, 'binding:vnic_type': 'macvtap', 'binding:profile': {'foo': 'bar'}, 'binding:host_id': 'binding-host-id-xxxx', 'description': 'newDescription', 'name': 'newName'}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_invalid_json_binding_profile(self):
        arglist = ['--binding-profile', '{"parent_name"}', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_set_port_invalid_key_value_binding_profile(self):
        arglist = ['--binding-profile', 'key', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_set_port_mixed_binding_profile(self):
        arglist = ['--binding-profile', 'foo=bar', '--binding-profile', '{"foo2": "bar2"}', self._port.name]
        verifylist = [('binding_profile', {'foo': 'bar', 'foo2': 'bar2'}), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'binding:profile': {'foo': 'bar', 'foo2': 'bar2'}}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_security_group(self):
        sg = network_fakes.FakeSecurityGroup.create_one_security_group()
        self.network_client.find_security_group = mock.Mock(return_value=sg)
        arglist = ['--security-group', sg.id, self._port.name]
        verifylist = [('security_group', [sg.id]), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'security_group_ids': [sg.id]}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_security_group_append(self):
        sg_1 = network_fakes.FakeSecurityGroup.create_one_security_group()
        sg_2 = network_fakes.FakeSecurityGroup.create_one_security_group()
        sg_3 = network_fakes.FakeSecurityGroup.create_one_security_group()
        self.network_client.find_security_group = mock.Mock(side_effect=[sg_2, sg_3])
        _testport = network_fakes.create_one_port({'security_group_ids': [sg_1.id]})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--security-group', sg_2.id, '--security-group', sg_3.id, _testport.name]
        verifylist = [('security_group', [sg_2.id, sg_3.id]), ('port', _testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'security_group_ids': [sg_1.id, sg_2.id, sg_3.id]}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_security_group_clear(self):
        arglist = ['--no-security-group', self._port.name]
        verifylist = [('no_security_group', True), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'security_group_ids': []}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_security_group_replace(self):
        sg1 = network_fakes.FakeSecurityGroup.create_one_security_group()
        sg2 = network_fakes.FakeSecurityGroup.create_one_security_group()
        _testport = network_fakes.create_one_port({'security_group_ids': [sg1.id]})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        self.network_client.find_security_group = mock.Mock(return_value=sg2)
        arglist = ['--security-group', sg2.id, '--no-security-group', _testport.name]
        verifylist = [('security_group', [sg2.id]), ('no_security_group', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'security_group_ids': [sg2.id]}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_allowed_address_pair(self):
        arglist = ['--allowed-address', 'ip-address=192.168.1.123', self._port.name]
        verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.123'}]), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_append_allowed_address_pair(self):
        _testport = network_fakes.create_one_port({'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--allowed-address', 'ip-address=192.168.1.45', _testport.name]
        verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.45'}]), ('port', _testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'allowed_address_pairs': [{'ip_address': '192.168.1.123'}, {'ip_address': '192.168.1.45'}]}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_overwrite_allowed_address_pair(self):
        _testport = network_fakes.create_one_port({'allowed_address_pairs': [{'ip_address': '192.168.1.123'}]})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--allowed-address', 'ip-address=192.168.1.45', '--no-allowed-address', _testport.name]
        verifylist = [('allowed_address_pairs', [{'ip-address': '192.168.1.45'}]), ('no_allowed_address_pair', True), ('port', _testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'allowed_address_pairs': [{'ip_address': '192.168.1.45'}]}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_no_allowed_address_pairs(self):
        arglist = ['--no-allowed-address', self._port.name]
        verifylist = [('no_allowed_address_pair', True), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'allowed_address_pairs': []}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_extra_dhcp_option(self):
        arglist = ['--extra-dhcp-option', 'name=foo,value=bar', self._port.name]
        verifylist = [('extra_dhcp_options', [{'name': 'foo', 'value': 'bar'}]), ('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'extra_dhcp_opts': [{'opt_name': 'foo', 'opt_value': 'bar'}]}
        self.network_client.update_port.assert_called_once_with(self._port, **attrs)
        self.assertIsNone(result)

    def test_set_port_security_enabled(self):
        arglist = ['--enable-port-security', self._port.id]
        verifylist = [('enable_port_security', True), ('port', self._port.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.update_port.assert_called_once_with(self._port, **{'port_security_enabled': True})

    def test_set_port_security_disabled(self):
        arglist = ['--disable-port-security', self._port.id]
        verifylist = [('disable_port_security', True), ('port', self._port.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.update_port.assert_called_once_with(self._port, **{'port_security_enabled': False})

    def test_set_port_with_qos(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        _testport = network_fakes.create_one_port({'qos_policy_id': None})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--qos-policy', qos_policy.id, _testport.name]
        verifylist = [('qos_policy', qos_policy.id), ('port', _testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'qos_policy_id': qos_policy.id}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_data_plane_status(self):
        _testport = network_fakes.create_one_port({'data_plane_status': None})
        self.network_client.find_port = mock.Mock(return_value=_testport)
        arglist = ['--data-plane-status', 'ACTIVE', _testport.name]
        verifylist = [('data_plane_status', 'ACTIVE'), ('port', _testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'data_plane_status': 'ACTIVE'}
        self.network_client.update_port.assert_called_once_with(_testport, **attrs)
        self.assertIsNone(result)

    def test_set_port_invalid_data_plane_status_value(self):
        arglist = ['--data-plane-status', 'Spider-Man', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def _test_set_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['red', 'blue', 'green']
        else:
            arglist = ['--no-tag']
            verifylist = [('no_tag', True)]
            expected_args = []
        arglist.append(self._port.name)
        verifylist.append(('port', self._port.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_port.called)
        self.network_client.set_tags.assert_called_once_with(self._port, test_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)

    def _test_create_with_numa_affinity_policy(self, policy):
        arglist = ['--numa-policy-%s' % policy, self._port.id]
        verifylist = [('numa_policy_%s' % policy, True), ('port', self._port.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.update_port.assert_called_once_with(self._port, **{'numa_affinity_policy': policy})

    def test_create_with_numa_affinity_policy_required(self):
        self._test_create_with_numa_affinity_policy('required')

    def test_create_with_numa_affinity_policy_preferred(self):
        self._test_create_with_numa_affinity_policy('preferred')

    def test_create_with_numa_affinity_policy_legacy(self):
        self._test_create_with_numa_affinity_policy('legacy')

    def test_set_hints_invalid_json(self):
        arglist = ['--network', self._port.network_id, '--hint', 'invalid json', 'test-port']
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    def test_set_hints_invalid_alias(self):
        arglist = ['--hint', 'invalid-alias=value', 'test-port']
        verifylist = [('hint', {'invalid-alias': 'value'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_hints_invalid_value(self):
        arglist = ['--hint', 'ovs-tx-steering=invalid-value', 'test-port']
        verifylist = [('hint', {'ovs-tx-steering': 'invalid-value'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_hints_valid_alias_value(self):
        testport = network_fakes.create_one_port()
        self.network_client.find_port = mock.Mock(return_value=testport)
        self.network_client.find_extension = mock.Mock(return_value=['port-hints', 'port-hint-ovs-tx-steering'])
        arglist = ['--hint', 'ovs-tx-steering=hash', testport.name]
        verifylist = [('hint', {'ovs-tx-steering': 'hash'}), ('port', testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_port.assert_called_once_with(testport, **{'hints': {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}})
        self.assertIsNone(result)

    def test_set_hints_valid_json(self):
        testport = network_fakes.create_one_port()
        self.network_client.find_port = mock.Mock(return_value=testport)
        self.network_client.find_extension = mock.Mock(return_value=['port-hints', 'port-hint-ovs-tx-steering'])
        arglist = ['--hint', '{"openvswitch": {"other_config": {"tx-steering": "hash"}}}', testport.name]
        verifylist = [('hint', {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}), ('port', testport.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_port.assert_called_once_with(testport, **{'hints': {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}})
        self.assertIsNone(result)