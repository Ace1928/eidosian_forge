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
def test_server_get_live_state(self):
    return_server = self.fc.servers.list()[1]
    return_server.id = '5678'
    self.patchobject(nova.NovaClientPlugin, 'is_version_supported', return_value=False)
    server = self._create_test_server(return_server, 'get_live_state_stack')
    server.properties.data['networks'] = [{'network': 'public_id', 'fixed_ip': '5.6.9.8'}]
    public_net = dict(id='public_id', name='public')
    private_net = dict(id='private_id', name='private')
    iface0 = create_fake_iface(port='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', net='public', ip='5.6.9.8', mac='fa:16:3e:8c:33:aa')
    port0 = dict(id=iface0.port_id, network_id=iface0.net_id, mac_address=iface0.mac_addr, fixed_ips=iface0.fixed_ips)
    iface1 = create_fake_iface(port='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', net='public', ip='4.5.6.7', mac='fa:16:3e:8c:22:aa')
    port1 = dict(id=iface1.port_id, network_id=iface1.net_id, mac_address=iface1.mac_addr, fixed_ips=iface1.fixed_ips)
    iface2 = create_fake_iface(port='cccccccc-cccc-cccc-cccc-cccccccccccc', net='private', ip='10.13.12.13', mac='fa:16:3e:8c:44:cc')
    port2 = dict(id=iface2.port_id, network_id=iface2.net_id, mac_address=iface2.mac_addr, fixed_ips=iface2.fixed_ips)
    self.patchobject(return_server, 'interface_list', return_value=[iface0, iface1, iface2])
    self.patchobject(neutronclient.Client, 'list_ports', return_value={'ports': [port0, port1, port2]})
    self.patchobject(neutronclient.Client, 'list_networks', side_effect=[{'networks': [public_net]}, {'networks': [public_net]}, {'networks': [private_net]}])
    self.patchobject(neutronclient.Client, 'list_floatingips', return_value={'floatingips': []})
    self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', side_effect=['public_id', 'private_id'])
    reality = server.get_live_state(server.properties.data)
    expected = {'flavor': '1', 'image': '2', 'name': 'sample-server2', 'networks': [{'fixed_ip': '4.5.6.7', 'network': 'public', 'port': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'}, {'fixed_ip': '5.6.9.8', 'network': 'public', 'port': None}, {'fixed_ip': '10.13.12.13', 'network': 'private', 'port': 'cccccccc-cccc-cccc-cccc-cccccccccccc'}], 'metadata': {}}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    expected_nets = expected.pop('networks')
    reality_nets = reality.pop('networks')
    for net in reality_nets:
        for exp_net in expected_nets:
            if net == exp_net:
                for key in net:
                    self.assertEqual(exp_net[key], net[key])
                break
    for key in reality.keys():
        self.assertEqual(expected[key], reality[key])