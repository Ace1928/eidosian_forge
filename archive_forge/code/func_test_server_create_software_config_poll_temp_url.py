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
def test_server_create_software_config_poll_temp_url(self):
    metadata_url, server = self._server_create_software_config_poll_temp_url()
    self.assertEqual({'os-collect-config': {'request': {'metadata_url': metadata_url}, 'collectors': ['request', 'local']}, 'deployments': []}, server.metadata_get())