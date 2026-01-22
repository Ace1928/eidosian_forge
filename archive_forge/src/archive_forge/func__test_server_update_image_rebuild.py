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
def _test_server_update_image_rebuild(self, status, policy='REBUILD', password=None):
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    server = self._create_test_server(return_server, 'srv_updimgrbld')
    new_image = 'F17-x86_64-gold'
    before_props = self.server_props.copy()
    before_props['image_update_policy'] = policy
    server.t = server.t.freeze(properties=before_props)
    server.reparse()
    update_props = before_props.copy()
    update_props['image'] = new_image
    if password:
        update_props['admin_pass'] = password
    update_template = server.t.freeze(properties=update_props)
    mock_rebuild = self.patchobject(self.fc.servers, 'rebuild')

    def get_sideeff(stat):

        def sideeff(*args):
            return_server.status = stat
            return return_server
        return sideeff
    for stat in status:
        self.patchobject(self.fc.servers, 'get', side_effect=get_sideeff(stat))
    scheduler.TaskRunner(server.update, update_template)()
    self.assertEqual((server.UPDATE, server.COMPLETE), server.state)
    mock_rebuild.assert_called_once()
    self.assertEqual((return_server, '2'), mock_rebuild.call_args.args)
    if 'REBUILD' == policy:
        self.assertLessEqual({'password': password, 'preserve_ephemeral': False, 'meta': {}, 'files': {}}.items(), mock_rebuild.call_args.kwargs.items())
    else:
        self.assertLessEqual({'password': password, 'preserve_ephemeral': True, 'meta': {}, 'files': {}}.items(), mock_rebuild.call_args.kwargs.items())