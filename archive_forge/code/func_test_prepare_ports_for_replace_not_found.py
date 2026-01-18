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