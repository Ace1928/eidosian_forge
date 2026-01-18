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