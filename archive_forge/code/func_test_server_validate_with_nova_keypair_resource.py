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
def test_server_validate_with_nova_keypair_resource(self):
    stack_name = 'srv_val_test'
    nova_keypair_template = '\n{\n  "AWSTemplateFormatVersion" : "2010-09-09",\n  "Description" : "WordPress",\n  "Resources" : {\n    "WebServer": {\n      "Type": "OS::Nova::Server",\n      "Properties": {\n        "image" : "F17-x86_64-gold",\n        "flavor"   : "m1.large",\n        "key_name"        : { "Ref": "SSHKey" },\n        "user_data"       : "wordpress"\n      }\n    },\n    "SSHKey": {\n      "Type": "OS::Nova::KeyPair",\n      "Properties": {\n        "name": "my_key"\n      }\n    }\n  }\n}\n'
    t = template_format.parse(nova_keypair_template)
    templ = template.Template(t)
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    stack = parser.Stack(utils.dummy_context(), stack_name, templ, stack_id=uuidutils.generate_uuid())
    resource_defns = templ.resource_definitions(stack)
    server = servers.Server('server_validate_test', resource_defns['WebServer'], stack)
    self.patchobject(glance.GlanceClientPlugin, 'get_image', return_value=self.mock_image)
    self.patchobject(nova.NovaClientPlugin, 'get_flavor', return_value=self.mock_flavor)
    self.assertIsNone(server.validate())