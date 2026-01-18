from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from urllib import parse as urlparse
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine.resources.openstack.heat import deployed_server
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_server_software_config_poll_heat(self):
    stack, server = self._server_create_software_config_poll_heat()
    md = {'os-collect-config': {'heat': {'auth_url': 'http://server.test:5000/v2.0', 'password': server.password, 'project_id': '8888', 'region_name': 'RegionOne', 'resource_name': 'server', 'stack_id': 'server_heat_s/%s' % stack.id, 'user_id': '1234'}, 'collectors': ['heat', 'local']}, 'deployments': []}
    self.assertEqual(md, server.metadata_get())
    md1 = {'os-collect-config': {'polling_interval': 10}}
    server.stack.t.t['resources']['server']['metadata'] = md1
    resource_defns = server.stack.t.resource_definitions(server.stack)
    scheduler.TaskRunner(server.update, resource_defns['server'])()
    occ = md['os-collect-config']
    occ.update(md1['os-collect-config'])
    self.assertEqual(md, server.metadata_get())