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
def test_server_update_flavor_resize_has_not_started(self):
    """Test update of server flavor if server resize has not started.

        Server resize is asynchronous operation in nova. So when heat is
        requesting resize and polling the server then the server may still be
        in ACTIVE state. So we need to wait some amount of time till the server
        status becomes RESIZE.
        """
    server = self.fc.servers.list()[1]
    server.id = '1234'
    server_resource = self._create_test_server(server, 'resize_server')
    update_props = self.server_props.copy()
    update_props['flavor'] = 'm1.small'
    update_template = server_resource.t.freeze(properties=update_props)
    self.patchobject(self.fc.servers, 'get', side_effect=ServerStatus(server, ['ACTIVE', 'ACTIVE', 'RESIZE', 'VERIFY_RESIZE', 'VERIFY_RESIZE', 'ACTIVE']))
    mock_post = self.patchobject(self.fc.client, 'post_servers_1234_action', return_value=(202, None))
    scheduler.TaskRunner(server_resource.update, update_template)()
    self.assertEqual((server_resource.UPDATE, server_resource.COMPLETE), server_resource.state)
    mock_post.assert_has_calls([mock.call(body={'resize': {'flavorRef': '2'}}), mock.call(body={'confirmResize': None})])