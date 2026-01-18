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
@mock.patch.object(heat_plugin.HeatClientPlugin, 'url_for')
def test_server_update_metadata_software_config_merge(self, fake_url):
    md = {'os-collect-config': {'polling_interval': 10}}
    fake_url.return_value = 'http://ip/heat-api-cfn/v1'
    server, ud_tmpl = self._server_create_software_config(stack_name='update_meta_sc', ret_tmpl=True, md=md)
    expected_md = {'os-collect-config': {'cfn': {'access_key_id': '4567', 'metadata_url': 'http://ip/heat-api-cfn/v1/', 'path': 'WebServer.Metadata', 'secret_access_key': '8901', 'stack_name': 'update_meta_sc'}, 'collectors': ['ec2', 'cfn', 'local'], 'polling_interval': 10}, 'deployments': []}
    self.assertEqual(expected_md, server.metadata_get())
    ud_tmpl.t['Resources']['WebServer']['Metadata'] = {'test': 123}
    resource_defns = ud_tmpl.resource_definitions(server.stack)
    scheduler.TaskRunner(server.update, resource_defns['WebServer'])()
    expected_md.update({'test': 123})
    self.assertEqual(expected_md, server.metadata_get())
    server.metadata_update()
    self.assertEqual(expected_md, server.metadata_get())