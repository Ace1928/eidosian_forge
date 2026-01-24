import collections
import contextlib
import copy
from unittest import mock
from keystoneauth1 import exceptions as ks_exceptions
from neutronclient.v2_0 import client as neutronclient
from novaclient import exceptions as nova_exceptions
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import requests
from urllib import parse as urlparse
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.nova import server as servers
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class ServerInternalPortTest(ServersTest):

    def setUp(self):
        super(ServerInternalPortTest, self).setUp()
        self.resolve = self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id')
        self.port_create = self.patchobject(neutronclient.Client, 'create_port')
        self.port_delete = self.patchobject(neutronclient.Client, 'delete_port')
        self.port_show = self.patchobject(neutronclient.Client, 'show_port')

        def neutron_side_effect(*args):
            if args[0] == 'subnet':
                return '1234'
            if args[0] == 'network':
                return '4321'
            if args[0] == 'port':
                return '12345'
        self.resolve.side_effect = neutron_side_effect

    def _return_template_stack_and_rsrc_defn(self, stack_name, temp):
        templ = template.Template(template_format.parse(temp), env=environment.Environment({'key_name': 'test'}))
        stack = parser.Stack(utils.dummy_context(), stack_name, templ, stack_id=uuidutils.generate_uuid(), stack_user_project_id='8888')
        resource_defns = templ.resource_definitions(stack)
        server = servers.Server('server', resource_defns['server'], stack)
        return (templ, stack, server)

    def test_build_nics_without_internal_port(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - port: 12345\n                  network: 4321\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        create_internal_port = self.patchobject(server, '_create_internal_port', return_value='12345')
        networks = [{'port': '12345', 'network': '4321'}]
        nics = server._build_nics(networks)
        self.assertEqual([{'port-id': '12345', 'net-id': '4321'}], nics)
        self.assertEqual(0, create_internal_port.call_count)

    def test_validate_internal_port_subnet_not_this_network(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        networks = server.properties['networks']
        for network in networks:
            server._validate_network(network)
        self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='not_this_network')
        ex = self.assertRaises(exception.StackValidationFailed, server._build_nics, networks)
        self.assertEqual('Specified subnet 1234 does not belongs to network 4321.', str(ex))

    def test_build_nics_create_internal_port_all_props_without_extras(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              security_groups:\n                - test_sec\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        self.patchobject(server, '_validate_belonging_subnet_to_net')
        self.patchobject(neutron.NeutronClientPlugin, 'get_secgroup_uuids', return_value=['5566'])
        self.port_create.return_value = {'port': {'id': '111222'}}
        data_set = self.patchobject(resource.Resource, 'data_set')
        network = [{'network': '4321', 'subnet': '1234', 'fixed_ip': '127.0.0.1'}]
        security_groups = ['test_sec']
        server._build_nics(network, security_groups)
        self.port_create.assert_called_once_with({'port': {'name': 'server-port-0', 'network_id': '4321', 'fixed_ips': [{'ip_address': '127.0.0.1', 'subnet_id': '1234'}], 'security_groups': ['5566']}})
        data_set.assert_called_once_with('internal_ports', '[{"id": "111222"}]')

    def test_build_nics_do_not_create_internal_port(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        self.port_create.return_value = {'port': {'id': '111222'}}
        data_set = self.patchobject(resource.Resource, 'data_set')
        network = [{'network': '4321'}]
        server._build_nics(network)
        self.assertFalse(self.port_create.called)
        self.assertFalse(data_set.called)

    def test_prepare_port_kwargs_with_extras(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                  port_extra_properties:\n                    mac_address: 00:00:00:00:00:00\n                    allowed_address_pairs:\n                      - ip_address: 127.0.0.1\n                        mac_address: None\n                      - mac_address: 00:00:00:00:00:00\n\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        network = {'network': '4321', 'subnet': '1234', 'fixed_ip': '127.0.0.1', 'port_extra_properties': {'value_specs': {}, 'mac_address': '00:00:00:00:00:00', 'allowed_address_pairs': [{'ip_address': '127.0.0.1', 'mac_address': None}, {'mac_address': '00:00:00:00:00:00'}]}}
        sec_uuids = ['8d94c72093284da88caaef5e985d96f7']
        self.patchobject(neutron.NeutronClientPlugin, 'get_secgroup_uuids', return_value=sec_uuids)
        kwargs = server._prepare_internal_port_kwargs(network, security_groups=['test_sec'])
        self.assertEqual({'network_id': '4321', 'security_groups': sec_uuids, 'fixed_ips': [{'ip_address': '127.0.0.1', 'subnet_id': '1234'}], 'mac_address': '00:00:00:00:00:00', 'allowed_address_pairs': [{'ip_address': '127.0.0.1'}, {'mac_address': '00:00:00:00:00:00'}]}, kwargs)

    def test_build_nics_create_internal_port_without_net(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - subnet: 1234\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='4321')
        net = {'subnet': '1234'}
        net_id = server._get_network_id(net)
        self.assertEqual('4321', net_id)
        self.assertEqual({'subnet': '1234'}, net)
        self.port_create.return_value = {'port': {'id': '111222'}}
        data_set = self.patchobject(resource.Resource, 'data_set')
        network = [{'subnet': '1234'}]
        server._build_nics(network)
        self.port_create.assert_called_once_with({'port': {'name': 'server-port-0', 'network_id': '4321', 'fixed_ips': [{'subnet_id': '1234'}]}})
        data_set.assert_called_once_with('internal_ports', '[{"id": "111222"}]')

    def test_calculate_networks_internal_ports(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                - port: 3344\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        data_mock = self.patchobject(server, '_data_get_ports')
        data_mock.side_effect = [[{'id': '1122'}], [{'id': '1122'}], []]
        self.port_create.return_value = {'port': {'id': '7788'}}
        data_set = self.patchobject(resource.Resource, 'data_set')
        old_net = [self.create_old_net(net='4321', subnet='1234', ip='127.0.0.1'), self.create_old_net(port='3344')]
        new_net = [{'port': '3344'}, {'port': '5566'}, {'network': '4321', 'subnet': '5678', 'fixed_ip': '10.0.0.1'}]
        interfaces = [create_fake_iface(port='1122', net='4321', ip='127.0.0.1', subnet='1234'), create_fake_iface(port='3344', net='4321', ip='10.0.0.2', subnet='subnet')]
        server.calculate_networks(old_net, new_net, interfaces)
        self.port_delete.assert_called_once_with('1122')
        self.port_create.assert_called_once_with({'port': {'name': 'server-port-1', 'network_id': '4321', 'fixed_ips': [{'subnet_id': '5678', 'ip_address': '10.0.0.1'}]}})
        self.assertEqual(2, data_set.call_count)
        data_set.assert_has_calls((mock.call('internal_ports', '[]'), mock.call('internal_ports', '[{"id": "7788"}]')))

    def test_calculate_networks_internal_ports_with_fipa(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                  floating_ip: 1199\n                - network: 8928\n                  subnet: 5678\n                  fixed_ip: 127.0.0.2\n                  floating_ip: 9911\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        self.patchobject(server, 'update_networks_matching_iface_port')
        server._data = {'internal_ports': '[{"id": "1122"}]'}
        self.port_create.return_value = {'port': {'id': '5566'}}
        self.patchobject(resource.Resource, 'data_set')
        self.resolve.side_effect = ['0912', '9021']
        fipa = self.patchobject(neutronclient.Client, 'update_floatingip', side_effect=[neutronclient.exceptions.NotFound, '9911', '11910', '1199'])
        old_net = [self.create_old_net(net='4321', subnet='1234', ip='127.0.0.1', port='1122', floating_ip='1199'), self.create_old_net(net='8928', subnet='5678', ip='127.0.0.2', port='3344', floating_ip='9911')]
        interfaces = [create_fake_iface(port='1122', net='4321', ip='127.0.0.1', subnet='1234'), create_fake_iface(port='3344', net='8928', ip='127.0.0.2', subnet='5678')]
        new_net = [{'network': '8928', 'subnet': '5678', 'fixed_ip': '127.0.0.2', 'port': '3344', 'floating_ip': '11910'}, {'network': '0912', 'subnet': '9021', 'fixed_ip': '127.0.0.1', 'floating_ip': '1199', 'port': '1122'}]
        server.calculate_networks(old_net, new_net, interfaces)
        fipa.assert_has_calls((mock.call('1199', {'floatingip': {'port_id': None}}), mock.call('9911', {'floatingip': {'port_id': None}}), mock.call('11910', {'floatingip': {'port_id': '3344', 'fixed_ip_address': '127.0.0.2'}}), mock.call('1199', {'floatingip': {'port_id': '1122', 'fixed_ip_address': '127.0.0.1'}})))

    def test_delete_fipa_with_exception_not_found_neutron(self):
        tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                  floating_ip: 1199\n                - network: 8928\n                  subnet: 5678\n                  fixed_ip: 127.0.0.2\n                  floating_ip: 9911\n        '
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
        delete_flip = mock.MagicMock(side_effect=[neutron.exceptions.NotFound(404)])
        server.client('neutron').update_floatingip = delete_flip
        self.assertIsNone(server._floating_ip_disassociate('flip123'))
        self.assertEqual(1, delete_flip.call_count)

    def test_delete_internal_ports(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        get_data = [{'internal_ports': '[{"id": "1122"}, {"id": "3344"}, {"id": "5566"}]'}, {'internal_ports': '[{"id": "1122"}, {"id": "3344"}, {"id": "5566"}]'}, {'internal_ports': '[{"id": "3344"}, {"id": "5566"}]'}, {'internal_ports': '[{"id": "5566"}]'}]
        self.patchobject(server, 'data', side_effect=get_data)
        data_set = self.patchobject(server, 'data_set')
        data_delete = self.patchobject(server, 'data_delete')
        server._delete_internal_ports()
        self.assertEqual(3, self.port_delete.call_count)
        self.assertEqual(('1122',), self.port_delete.call_args_list[0][0])
        self.assertEqual(('3344',), self.port_delete.call_args_list[1][0])
        self.assertEqual(('5566',), self.port_delete.call_args_list[2][0])
        self.assertEqual(3, data_set.call_count)
        data_set.assert_has_calls((mock.call('internal_ports', '[{"id": "3344"}, {"id": "5566"}]'), mock.call('internal_ports', '[{"id": "5566"}]'), mock.call('internal_ports', '[]')))
        data_delete.assert_called_once_with('internal_ports')

    def test_get_data_internal_ports(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        server._data = {'internal_ports': '[{"id": "1122"}]'}
        data = server._data_get_ports()
        self.assertEqual([{'id': '1122'}], data)
        server._data = {'internal_ports': ''}
        data = server._data_get_ports()
        self.assertEqual([], data)

    def test_store_external_ports(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)

        class Fake(object):

            def interface_list(self):
                return [iface('1122'), iface('1122'), iface('2233'), iface('3344')]
        server.client = mock.Mock()
        server.client().servers.get.return_value = Fake()
        server.client_plugin = mock.Mock()
        server._data = {'internal_ports': '[{"id": "1122"}]', 'external_ports': '[{"id": "3344"},{"id": "5566"}]'}
        iface = collections.namedtuple('iface', ['port_id'])
        update_data = self.patchobject(server, '_data_update_ports')
        server.store_external_ports()
        self.assertEqual(2, update_data.call_count)
        self.assertEqual(('5566', 'delete'), update_data.call_args_list[0][0])
        self.assertEqual({'port_type': 'external_ports'}, update_data.call_args_list[0][1])
        self.assertEqual(('2233', 'add'), update_data.call_args_list[1][0])
        self.assertEqual({'port_type': 'external_ports'}, update_data.call_args_list[1][1])

    def test_prepare_ports_for_replace_detach_failed(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)

        class Fake(object):

            def interface_list(self):
                return [iface(1122)]
        iface = collections.namedtuple('iface', ['port_id'])
        server.resource_id = 'ser-11'
        port_ids = [{'id': 1122}]
        server._data = {'internal_ports': jsonutils.dumps(port_ids)}
        self.patchobject(nova.NovaClientPlugin, 'client')
        self.patchobject(nova.NovaClientPlugin, 'interface_detach')
        self.patchobject(nova.NovaClientPlugin, 'fetch_server')
        self.patchobject(nova.NovaClientPlugin.check_interface_detach.retry, 'sleep')
        nova.NovaClientPlugin.fetch_server.side_effect = [Fake()] * 10
        exc = self.assertRaises(exception.InterfaceDetachFailed, server.prepare_for_replace)
        self.assertIn('Failed to detach interface (1122) from server (ser-11)', str(exc))

    def test_prepare_ports_for_replace(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        server.resource_id = 'test_server'
        port_ids = [{'id': '1122'}, {'id': '3344'}]
        external_port_ids = [{'id': '5566'}]
        server._data = {'internal_ports': jsonutils.dumps(port_ids), 'external_ports': jsonutils.dumps(external_port_ids)}
        self.patchobject(nova.NovaClientPlugin, 'client')
        nova_server = self.fc.servers.list()[1]
        server.client().servers.get.return_value = nova_server
        self.patchobject(nova.NovaClientPlugin, 'interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
        server.prepare_for_replace()
        nova.NovaClientPlugin.interface_detach.assert_has_calls([mock.call('test_server', '1122'), mock.call('test_server', '3344'), mock.call('test_server', '5566')])

    def test_prepare_ports_for_replace_not_found(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        server.resource_id = 'test_server'
        port_ids = [{'id': '1122'}, {'id': '3344'}]
        external_port_ids = [{'id': '5566'}]
        server._data = {'internal_ports': jsonutils.dumps(port_ids), 'external_ports': jsonutils.dumps(external_port_ids)}
        self.patchobject(nova.NovaClientPlugin, 'client')
        self.patchobject(nova.NovaClientPlugin, 'fetch_server', side_effect=nova_exceptions.NotFound(404))
        check_detach = self.patchobject(nova.NovaClientPlugin, 'check_interface_detach')
        self.patchobject(nova.NovaClientPlugin, 'client')
        nova_server = self.fc.servers.list()[1]
        nova_server.status = 'DELETED'
        server.client().servers.get.return_value = nova_server
        server.prepare_for_replace()
        self.assertEqual(3, check_detach.call_count)
        self.assertEqual(0, self.port_delete.call_count)

    def test_prepare_ports_for_replace_error_state(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        server.resource_id = 'test_server'
        port_ids = [{'id': '1122'}, {'id': '3344'}]
        external_port_ids = [{'id': '5566'}]
        server._data = {'internal_ports': jsonutils.dumps(port_ids), 'external_ports': jsonutils.dumps(external_port_ids)}
        self.patchobject(nova.NovaClientPlugin, 'client')
        nova_server = self.fc.servers.list()[1]
        nova_server.status = 'ERROR'
        server.client().servers.get.return_value = nova_server
        self.patchobject(nova.NovaClientPlugin, 'interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
        data_set = self.patchobject(server, 'data_set')
        data_delete = self.patchobject(server, 'data_delete')
        server.prepare_for_replace()
        self.assertEqual(2, self.port_delete.call_count)
        self.assertEqual(('1122',), self.port_delete.call_args_list[0][0])
        self.assertEqual(('3344',), self.port_delete.call_args_list[1][0])
        data_set.assert_has_calls((mock.call('internal_ports', '[{"id": "3344"}]'), mock.call('internal_ports', '[{"id": "1122"}]')))
        data_delete.assert_called_once_with('internal_ports')

    def test_prepare_ports_for_replace_not_created(self):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        prepare_mock = self.patchobject(server, 'prepare_ports_for_replace')
        server.prepare_for_replace()
        self.assertIsNone(server.resource_id)
        self.assertEqual(0, prepare_mock.call_count)

    @mock.patch.object(server_network_mixin.ServerNetworkMixin, 'store_external_ports')
    def test_restore_ports_after_rollback(self, store_ports):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        server.resource_id = 'existing_server'
        port_ids = [{'id': 1122}, {'id': 3344}]
        external_port_ids = [{'id': 5566}]
        server._data = {'internal_ports': jsonutils.dumps(port_ids), 'external_ports': jsonutils.dumps(external_port_ids)}
        self.patchobject(nova.NovaClientPlugin, '_check_active')
        nova.NovaClientPlugin._check_active.side_effect = [False, True]
        old_server = mock.Mock()
        old_server.resource_id = 'old_server'
        stack._backup_stack = mock.Mock()
        stack._backup_stack().resources.get.return_value = old_server
        old_server._data_get_ports.side_effect = [port_ids, external_port_ids]
        self.patchobject(nova.NovaClientPlugin, 'interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'interface_attach')
        self.patchobject(nova.NovaClientPlugin, 'check_interface_attach', return_value=True)
        server.restore_prev_rsrc()
        self.assertEqual(2, nova.NovaClientPlugin._check_active.call_count)
        nova.NovaClientPlugin.interface_detach.assert_has_calls([mock.call('existing_server', 1122), mock.call('existing_server', 3344), mock.call('existing_server', 5566)])
        nova.NovaClientPlugin.interface_attach.assert_has_calls([mock.call('old_server', 1122), mock.call('old_server', 3344), mock.call('old_server', 5566)])

    @mock.patch.object(server_network_mixin.ServerNetworkMixin, 'store_external_ports')
    def test_restore_ports_after_rollback_attach_failed(self, store_ports):
        t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl_server_with_network_id)
        server.resource_id = 'existing_server'
        port_ids = [{'id': 1122}, {'id': 3344}]
        server._data = {'internal_ports': jsonutils.dumps(port_ids)}
        self.patchobject(nova.NovaClientPlugin, '_check_active')
        nova.NovaClientPlugin._check_active.return_value = True
        old_server = mock.Mock()
        old_server.resource_id = 'old_server'
        stack._backup_stack = mock.Mock()
        stack._backup_stack().resources.get.return_value = old_server
        old_server._data_get_ports.side_effect = [port_ids, []]

        class Fake(object):

            def interface_list(self):
                return [iface(1122)]
        iface = collections.namedtuple('iface', ['port_id'])
        self.patchobject(nova.NovaClientPlugin, 'interface_detach')
        self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'interface_attach')
        self.patchobject(nova.NovaClientPlugin, 'fetch_server')
        self.patchobject(nova.NovaClientPlugin.check_interface_attach.retry, 'sleep')
        nova.NovaClientPlugin.fetch_server.side_effect = [Fake()] * 11
        exc = self.assertRaises(exception.InterfaceAttachFailed, server.restore_prev_rsrc)
        self.assertIn('Failed to attach interface (3344) to server (old_server)', str(exc))

    @mock.patch.object(server_network_mixin.ServerNetworkMixin, 'store_external_ports')
    def test_restore_ports_after_rollback_convergence(self, store_ports):
        t = template_format.parse(tmpl_server_with_network_id)
        stack = utils.parse_stack(t)
        stack.store()
        self.patchobject(nova.NovaClientPlugin, '_check_active')
        nova.NovaClientPlugin._check_active.return_value = True
        prev_rsrc = stack['server']
        prev_rsrc.state_set(prev_rsrc.UPDATE, prev_rsrc.COMPLETE)
        prev_rsrc.resource_id = 'prev_rsrc'
        resource_defns = stack.t.resource_definitions(stack)
        existing_rsrc = servers.Server('server', resource_defns['server'], stack)
        existing_rsrc.stack = stack
        existing_rsrc.current_template_id = stack.t.id
        existing_rsrc.resource_id = 'existing_rsrc'
        existing_rsrc.state_set(existing_rsrc.UPDATE, existing_rsrc.COMPLETE)
        port_ids = [{'id': 1122}, {'id': 3344}]
        external_port_ids = [{'id': 5566}]
        existing_rsrc.data_set('internal_ports', jsonutils.dumps(port_ids))
        existing_rsrc.data_set('external_ports', jsonutils.dumps(external_port_ids))
        prev_rsrc.replaced_by = existing_rsrc.id
        self.patchobject(nova.NovaClientPlugin, 'interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
        self.patchobject(nova.NovaClientPlugin, 'interface_attach')
        self.patchobject(nova.NovaClientPlugin, 'check_interface_attach', return_value=True)
        prev_rsrc.restore_prev_rsrc(convergence=True)
        nova.NovaClientPlugin.interface_detach.assert_has_calls([mock.call('existing_rsrc', 1122), mock.call('existing_rsrc', 3344), mock.call('existing_rsrc', 5566)])
        nova.NovaClientPlugin.interface_attach.assert_has_calls([mock.call('prev_rsrc', 1122), mock.call('prev_rsrc', 3344), mock.call('prev_rsrc', 5566)])