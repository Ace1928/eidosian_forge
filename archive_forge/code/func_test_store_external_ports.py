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