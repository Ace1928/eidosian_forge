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
def test_server_update_subnet_to_network_with_security_group(self):
    return_server = self.fc.servers.list()[3]
    return_server.id = '9102'
    server = self._create_test_server(return_server, 'update_subnet')
    before_props = self.server_props.copy()
    before_props['networks'] = [{'subnet': 'aaa09d50-8c23-4498-a542-aa0deb24f73e'}]
    before_props['security_groups'] = ['the_sg']
    new_networks = [{'network': '2a60cbaa-3d33-4af6-a9ce-83594ac546fc'}]
    update_props = self.server_props.copy()
    update_props['networks'] = new_networks
    update_props['security_groups'] = ['the_sg']
    update_template = server.t.freeze(properties=update_props)
    server.t = server.t.freeze(properties=before_props)
    sec_uuids = ['86c0f8ae-23a8-464f-8603-c54113ef5467']
    self.patchobject(self.fc.servers, 'get', return_value=return_server)
    self.patchobject(neutron.NeutronClientPlugin, 'get_secgroup_uuids', return_value=sec_uuids)
    self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='05d8e681-4b37-4570-bc8d-810089f706b2')
    iface = create_fake_iface(port='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', net='05d8e681-4b37-4570-bc8d-810089f706b2', subnet='aaa09d50-8c23-4498-a542-aa0deb24f73e', ip='1.2.3.4')
    self.patchobject(return_server, 'interface_list', return_value=[iface])
    mock_detach = self.patchobject(return_server, 'interface_detach')
    mock_attach = self.patchobject(return_server, 'interface_attach')

    def interface_attach_mock(port, net):

        class attachment(object):

            def __init__(self, port_id, net_id):
                self.port_id = port_id
                self.net_id = net_id
        return attachment(port, net)
    mock_attach.return_value = interface_attach_mock('ad4a231b-67f7-45fe-aee9-461176b48203', '2a60cbaa-3d33-4af6-a9ce-83594ac546fc')
    mock_detach_check = self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
    mock_attach_check = self.patchobject(nova.NovaClientPlugin, 'check_interface_attach', return_value=True)
    mock_update_port = self.patchobject(neutronclient.Client, 'update_port')
    scheduler.TaskRunner(server.update, update_template, before=server.t)()
    self.assertEqual((server.UPDATE, server.COMPLETE), server.state)
    self.assertEqual(1, mock_detach.call_count)
    self.assertEqual(1, mock_attach.call_count)
    self.assertEqual(1, mock_detach_check.call_count)
    self.assertEqual(1, mock_attach_check.call_count)
    kwargs = {'security_groups': sec_uuids}
    mock_update_port.assert_called_with('ad4a231b-67f7-45fe-aee9-461176b48203', {'port': kwargs})