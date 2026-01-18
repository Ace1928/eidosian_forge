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
def test_server_without_ip_address(self):
    return_server = self.fc.servers.list()[3]
    return_server.id = '9102'
    server = self._create_test_server(return_server, 'wo_ipaddr')
    self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value=None)
    self.patchobject(neutronclient.Client, 'list_ports', return_value={'ports': [{'id': 'p_id', 'name': 'p_name', 'fixed_ips': [], 'network_id': 'n_id'}]})
    self.patchobject(neutronclient.Client, 'list_networks', return_value={'networks': [{'id': 'n_id', 'name': 'empty_net'}]})
    self.patchobject(self.fc.servers, 'get', return_value=return_server)
    self.patchobject(return_server, 'interface_list', return_value=[])
    mock_detach = self.patchobject(return_server, 'interface_detach')
    mock_attach = self.patchobject(return_server, 'interface_attach')
    self.assertEqual({'empty_net': []}, server.FnGetAtt('addresses'))
    self.assertEqual({'empty_net': []}, server.FnGetAtt('networks'))
    self.assertEqual(0, mock_detach.call_count)
    self.assertEqual(0, mock_attach.call_count)