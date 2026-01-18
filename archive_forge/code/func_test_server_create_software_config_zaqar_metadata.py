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
def test_server_create_software_config_zaqar_metadata(self):
    md = {'os-collect-config': {'polling_interval': 10}}
    queue_id, server = self._server_create_software_config_zaqar(md=md)
    self.assertEqual({'os-collect-config': {'zaqar': {'user_id': '1234', 'password': server.password, 'auth_url': 'http://server.test:5000/v2.0', 'project_id': '8888', 'region_name': 'RegionOne', 'queue_id': queue_id}, 'collectors': ['zaqar', 'local'], 'polling_interval': 10}, 'deployments': []}, server.metadata_get())