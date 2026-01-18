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
def test_server_properties_validation_create_and_update_fail(self):
    return_server = self.fc.servers.list()[1]
    server = self._create_test_server(return_server, 'my_server')
    ex = glance.client_exception.EntityMatchNotFound(entity='image', args='Update Image')
    self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', side_effect=[1, ex])
    update_props = self.server_props.copy()
    update_props['image'] = 'Update Image'
    update_template = server.t.freeze(properties=update_props)
    updater = scheduler.TaskRunner(server.update, update_template)
    err = self.assertRaises(exception.ResourceFailure, updater)
    self.assertEqual("StackValidationFailed: resources.my_server: Property error: Properties.image: Error validating value '1': No image matching Update Image.", str(err))