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
def test_server_create_metadata(self):
    stack_name = 'create_metadata_test_stack'
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    return_server = self.fc.servers.list()[1]
    tmpl, stack = self._setup_test_stack(stack_name)
    tmpl['Resources']['WebServer']['Properties']['metadata'] = {'a': 1}
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('create_metadata_test_server', resource_defns['WebServer'], stack)
    self.patchobject(server, 'store_external_ports')
    mock_create = self.patchobject(self.fc.servers, 'create', return_value=return_server)
    scheduler.TaskRunner(server.create)()
    args, kwargs = mock_create.call_args
    self.assertEqual({'a': '1'}, kwargs['meta'])