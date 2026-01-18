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
def test_server_create_with_subnet_security_group(self):
    stack_name = 'server_with_subnet_security_group'
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    return_server = self.fc.servers.list()[1]
    tmpl, stack = self._setup_test_stack(stack_name, test_templ=tmpl_server_with_sub_secu_group)
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('server_with_sub_secu', resource_defns['server'], stack)
    mock_find = self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='2a60cbaa-3d33-4af6-a9ce-83594ac546fc')
    sec_uuids = ['86c0f8ae-23a8-464f-8603-c54113ef5467']
    self.patchobject(neutron.NeutronClientPlugin, 'get_secgroup_uuids', return_value=sec_uuids)
    self.patchobject(server, 'store_external_ports')
    self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='05d8e681-4b37-4570-bc8d-810089f706b2')
    mock_create_port = self.patchobject(neutronclient.Client, 'create_port')
    mock_create = self.patchobject(self.fc.servers, 'create', return_value=return_server)
    scheduler.TaskRunner(server.create)()
    kwargs = {'network_id': '05d8e681-4b37-4570-bc8d-810089f706b2', 'fixed_ips': [{'subnet_id': '2a60cbaa-3d33-4af6-a9ce-83594ac546fc'}], 'security_groups': sec_uuids, 'name': 'server_with_sub_secu-port-0'}
    mock_create_port.assert_called_with({'port': kwargs})
    self.assertEqual(1, mock_find.call_count)
    args, kwargs = mock_create.call_args
    self.assertEqual({}, kwargs['meta'])