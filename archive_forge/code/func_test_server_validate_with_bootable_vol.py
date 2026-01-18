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
def test_server_validate_with_bootable_vol(self):
    stack_name = 'srv_val_bootvol'
    tmpl, stack = self._setup_test_stack(stack_name)
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    self.stub_VolumeConstraint_validate()
    web_server = tmpl.t['Resources']['WebServer']
    del web_server['Properties']['image']

    def create_server(device_name):
        web_server['Properties']['block_device_mapping'] = [{'device_name': device_name, 'volume_id': '5d7e27da-6703-4f7e-9f94-1f67abef734c', 'delete_on_termination': False}]
        resource_defns = tmpl.resource_definitions(stack)
        server = servers.Server('server_with_bootable_volume', resource_defns['WebServer'], stack)
        return server
    server = create_server(u'vda')
    self.assertIsNone(server.validate())
    server = create_server('vda')
    self.assertIsNone(server.validate())
    server = create_server('vdb')
    ex = self.assertRaises(exception.StackValidationFailed, server.validate)
    self.assertEqual('Neither image nor bootable volume is specified for instance server_with_bootable_volume', str(ex))
    web_server['Properties']['image'] = ''
    server = create_server('vdb')
    self.assertIsNone(server.validate())