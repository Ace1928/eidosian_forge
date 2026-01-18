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
def test_server_image_name_err(self):
    stack_name = 'img_name_err'
    tmpl, stack = self._setup_test_stack(stack_name)
    mock_image = self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id')
    self.stub_KeypairConstraint_validate()
    mock_image.side_effect = glance.client_exception.EntityMatchNotFound(entity='image', args={'name': 'Slackware'})
    tmpl['Resources']['WebServer']['Properties']['image'] = 'Slackware'
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('WebServer', resource_defns['WebServer'], stack)
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(server.create))
    self.assertIn("No image matching {'name': 'Slackware'}.", str(error))