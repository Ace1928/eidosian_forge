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
def _test_check_snapshot_complete_with_task_state(self, task_state='active'):
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    server = self._create_test_server(return_server, 'test_server_snapshot')
    image = mock.MagicMock(status='active')
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=image)
    server_with_task_state = mock.Mock()
    setattr(server_with_task_state, 'OS-EXT-STS:task_state', task_state)
    mock_get = self.patchobject(nova.NovaClientPlugin, 'get_server', return_value=server_with_task_state)
    if task_state not in {'image_uploading', 'image_snapshot_pending', 'image_snapshot', 'image_pending_upload'}:
        self.assertTrue(server.check_snapshot_complete('fake_iamge_id'))
    else:
        self.assertFalse(server.check_snapshot_complete('fake_iamge_id'))
    mock_get.assert_called_once_with(server.resource_id)