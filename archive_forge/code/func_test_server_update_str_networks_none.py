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
def test_server_update_str_networks_none(self):
    return_server = self.fc.servers.list()[1]
    return_server.id = '5678'
    old_networks = [{'port': '95e25541-d26a-478d-8f36-ae1c8f6b74dc'}, {'port': '4121f61a-1b2e-4ab0-901e-eade9b1cb09d'}, {'network': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'fixed_ip': '31.32.33.34'}]
    server = self._create_test_server(return_server, 'networks_update', networks=old_networks)
    update_props = self.server_props.copy()
    update_props['networks'] = [{'allocate_network': 'none'}]
    update_template = server.t.freeze(properties=update_props)
    self.patchobject(self.fc.servers, 'get', return_value=return_server)
    port_interfaces = [create_fake_iface(port='95e25541-d26a-478d-8f36-ae1c8f6b74dc', net='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', ip='11.12.13.14'), create_fake_iface(port='4121f61a-1b2e-4ab0-901e-eade9b1cb09d', net='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', ip='21.22.23.24'), create_fake_iface(port='0907fa82-a024-43c2-9fc5-efa1bccaa74a', net='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', ip='31.32.33.34')]
    self.patchobject(return_server, 'interface_list', return_value=port_interfaces)
    mock_detach = self.patchobject(return_server, 'interface_detach')
    self.patchobject(nova.NovaClientPlugin, 'check_interface_detach', return_value=True)
    mock_attach = self.patchobject(return_server, 'interface_attach')
    scheduler.TaskRunner(server.update, update_template)()
    self.assertEqual((server.UPDATE, server.COMPLETE), server.state)
    self.assertEqual(3, mock_detach.call_count)
    self.assertEqual(0, mock_attach.call_count)