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
def test_snapshot_policy_image_failed(self):
    t = template_format.parse(wp_template)
    t['Resources']['WebServer']['DeletionPolicy'] = 'Snapshot'
    tmpl = template.Template(t)
    stack = parser.Stack(utils.dummy_context(), 'snapshot_policy', tmpl)
    stack.store()
    self.patchobject(stack['WebServer'], 'store_external_ports')
    mock_plugin = self.patchobject(nova.NovaClientPlugin, 'client')
    mock_plugin.return_value = self.fc
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=self.mock_image)
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    mock_create = self.patchobject(self.fc.servers, 'create')
    mock_create.return_value = return_server
    mock_get = self.patchobject(self.fc.servers, 'get')
    mock_get.return_value = return_server
    image = self.fc.servers.create_image('1234', 'name')
    create_image = self.patchobject(self.fc.servers, 'create_image')
    create_image.return_value = image
    delete_server = self.patchobject(self.fc.servers, 'delete')
    delete_server.side_effect = nova_exceptions.NotFound(404)
    scheduler.TaskRunner(stack.create)()
    self.assertEqual((stack.CREATE, stack.COMPLETE), stack.state)
    failed_image = mock.Mock(**{'id': 456, 'name': 'CentOS 5.2', 'updated': '2010-10-10T12:00:00Z', 'created': '2010-08-10T12:00:00Z', 'status': 'ERROR'})
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=failed_image)
    return_server = self.fc.servers.list()[1]
    scheduler.TaskRunner(stack.delete)()
    self.assertEqual((stack.DELETE, stack.FAILED), stack.state)
    self.assertEqual('Resource DELETE failed: Error: resources.WebServer: ERROR', stack.status_reason)
    create_image.assert_called_once_with('1234', utils.PhysName('snapshot_policy', 'WebServer'))
    delete_server.assert_not_called()