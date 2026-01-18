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
def test_server_create_with_stack_scheduler_hints(self):
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    return_server = self.fc.servers.list()[1]
    return_server.id = '5678'
    sh.cfg.CONF.set_override('stack_scheduler_hints', True)
    stack_name = 'test_server_w_stack_sched_hints_s'
    server_name = 'server_w_stack_sched_hints'
    t, stack = self._get_test_template(stack_name, server_name)
    self.patchobject(stack, 'path_in_stack', return_value=[('parent', stack.name)])
    resource_defns = t.resource_definitions(stack)
    server = servers.Server(server_name, resource_defns['WebServer'], stack)
    self.patchobject(server, 'store_external_ports')
    stack.add_resource(server)
    self.assertIsNotNone(server.uuid)
    mock_create = self.patchobject(self.fc.servers, 'create', return_value=return_server)
    shm = sh.SchedulerHintsMixin
    scheduler_hints = {shm.HEAT_ROOT_STACK_ID: stack.root_stack_id(), shm.HEAT_STACK_ID: stack.id, shm.HEAT_STACK_NAME: stack.name, shm.HEAT_PATH_IN_STACK: [','.join(['parent', stack.name])], shm.HEAT_RESOURCE_NAME: server.name, shm.HEAT_RESOURCE_UUID: server.uuid}
    scheduler.TaskRunner(server.create)()
    _, kwargs = mock_create.call_args
    self.assertEqual(scheduler_hints, kwargs['scheduler_hints'])
    self.assertEqual({}, kwargs['meta'])
    self.assertGreater(server.id, 0)