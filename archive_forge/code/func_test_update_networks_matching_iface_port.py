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
def test_update_networks_matching_iface_port(self):
    return_server = self.fc.servers.list()[3]
    server = self._create_test_server(return_server, 'networks_update')
    nets = [self.create_old_net(port='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'), self.create_old_net(net='gggggggg-1111-1111-1111-gggggggggggg', ip='1.2.3.4'), self.create_old_net(net='gggggggg-1111-1111-1111-gggggggggggg'), self.create_old_net(port='dddddddd-dddd-dddd-dddd-dddddddddddd'), self.create_old_net(net='gggggggg-1111-1111-1111-gggggggggggg', ip='5.6.7.8'), self.create_old_net(net='gggggggg-1111-1111-1111-gggggggggggg', subnet='hhhhhhhh-1111-1111-1111-hhhhhhhhhhhh'), self.create_old_net(subnet='iiiiiiii-1111-1111-1111-iiiiiiiiiiii')]
    interfaces = [create_fake_iface(port='cccccccc-cccc-cccc-cccc-cccccccccccc', net=nets[2]['network'], ip='10.0.0.11'), create_fake_iface(port=nets[3]['port'], net='gggggggg-1111-1111-1111-gggggggggggg', ip='10.0.0.12'), create_fake_iface(port=nets[0]['port'], net='gggggggg-1111-1111-1111-gggggggggggg', ip='10.0.0.13'), create_fake_iface(port='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', net=nets[1]['network'], ip=nets[1]['fixed_ip']), create_fake_iface(port='eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee', net=nets[4]['network'], ip=nets[4]['fixed_ip']), create_fake_iface(port='gggggggg-gggg-gggg-gggg-gggggggggggg', net='gggggggg-1111-1111-1111-gggggggggggg', ip='10.0.0.14', subnet=nets[6]['subnet']), create_fake_iface(port='ffffffff-ffff-ffff-ffff-ffffffffffff', net=nets[5]['network'], ip='10.0.0.15', subnet=nets[5]['subnet'])]
    expected = [{'port': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'fixed_ip': '10.0.0.13', 'subnet': None, 'floating_ip': None, 'port_extra_properties': None, 'uuid': None, 'allocate_network': None, 'tag': None}, {'port': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'fixed_ip': '1.2.3.4', 'subnet': None, 'port_extra_properties': None, 'floating_ip': None, 'uuid': None, 'allocate_network': None, 'tag': None}, {'port': 'cccccccc-cccc-cccc-cccc-cccccccccccc', 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'fixed_ip': '10.0.0.11', 'subnet': None, 'port_extra_properties': None, 'floating_ip': None, 'uuid': None, 'allocate_network': None, 'tag': None}, {'port': 'dddddddd-dddd-dddd-dddd-dddddddddddd', 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'fixed_ip': '10.0.0.12', 'subnet': None, 'port_extra_properties': None, 'floating_ip': None, 'uuid': None, 'allocate_network': None, 'tag': None}, {'port': 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee', 'uuid': None, 'fixed_ip': '5.6.7.8', 'subnet': None, 'port_extra_properties': None, 'floating_ip': None, 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'allocate_network': None, 'tag': None}, {'port': 'ffffffff-ffff-ffff-ffff-ffffffffffff', 'uuid': None, 'fixed_ip': '10.0.0.15', 'subnet': 'hhhhhhhh-1111-1111-1111-hhhhhhhhhhhh', 'port_extra_properties': None, 'floating_ip': None, 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'allocate_network': None, 'tag': None}, {'port': 'gggggggg-gggg-gggg-gggg-gggggggggggg', 'uuid': None, 'fixed_ip': '10.0.0.14', 'subnet': 'iiiiiiii-1111-1111-1111-iiiiiiiiiiii', 'port_extra_properties': None, 'floating_ip': None, 'network': 'gggggggg-1111-1111-1111-gggggggggggg', 'allocate_network': None, 'tag': None}]
    self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='gggggggg-1111-1111-1111-gggggggggggg')
    self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='gggggggg-1111-1111-1111-gggggggggggg')
    server.update_networks_matching_iface_port(nets, interfaces)
    self.assertEqual(expected, nets)