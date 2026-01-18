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
def test_server_create_with_image_id(self):
    return_server = self.fc.servers.list()[1]
    return_server.id = '5678'
    server_name = 'test_server_create_image_id'
    server = self._setup_test_server(return_server, server_name, override_name=True)
    server.resource_id = '1234'
    interfaces = [create_fake_iface(port='1234', mac='fa:16:3e:8c:22:aa', ip='4.5.6.7'), create_fake_iface(port='5678', mac='fa:16:3e:8c:33:bb', ip='5.6.9.8'), create_fake_iface(port='1013', mac='fa:16:3e:8c:44:cc', ip='10.13.12.13')]
    ports = [dict(id=interfaces[0].port_id, mac_address=interfaces[0].mac_addr, fixed_ips=interfaces[0].fixed_ips, network_id='public_id'), dict(id=interfaces[1].port_id, mac_address=interfaces[1].mac_addr, fixed_ips=interfaces[1].fixed_ips, network_id='public_id'), dict(id=interfaces[2].port_id, mac_address=interfaces[2].mac_addr, fixed_ips=interfaces[2].fixed_ips, network_id='private_id')]
    public_net = dict(id='public_id', name='public')
    private_net = dict(id='private_id', name='private')
    self.patchobject(self.fc.servers, 'get', return_value=return_server)
    self.patchobject(neutronclient.Client, 'list_ports', return_value={'ports': ports})
    self.patchobject(neutronclient.Client, 'list_networks', side_effect=[{'networks': [public_net]}, {'networks': [public_net]}, {'networks': [private_net]}])
    self.patchobject(neutronclient.Client, 'list_floatingips', return_value={'floatingips': []})
    self.patchobject(return_server, 'interface_detach')
    self.patchobject(return_server, 'interface_attach')
    public_ip = return_server.networks['public'][0]
    self.assertEqual('1234', server.FnGetAtt('addresses')['public'][0]['port'])
    self.assertEqual('5678', server.FnGetAtt('addresses')['public'][1]['port'])
    self.assertEqual(public_ip, server.FnGetAtt('addresses')['public'][0]['addr'])
    self.assertEqual(public_ip, server.FnGetAtt('networks')['public'][0])
    private_ip = return_server.networks['private'][0]
    self.assertEqual('1013', server.FnGetAtt('addresses')['private'][0]['port'])
    self.assertEqual(private_ip, server.FnGetAtt('addresses')['private'][0]['addr'])
    self.assertEqual(private_ip, server.FnGetAtt('networks')['private'][0])
    self.assertEqual(server_name, server.FnGetAtt('name'))