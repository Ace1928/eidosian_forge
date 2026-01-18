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
def test_server_create_deployment_swift_data_bad_container(self):
    server_name = 'server'
    stack_name = '%s_s' % server_name
    tmpl, stack = self._setup_test_stack(stack_name, ds_deployment_data_bad_container_tmpl)
    props = tmpl.t['resources']['server']['properties']
    props['software_config_transport'] = 'POLL_TEMP_URL'
    self.server_props = props
    resource_defns = tmpl.resource_definitions(stack)
    server = deployed_server.DeployedServer(server_name, resource_defns[server_name], stack)
    self.assertRaises(exception.StackValidationFailed, server.validate)