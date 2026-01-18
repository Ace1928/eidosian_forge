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
def test_validate_internal_port_subnet_not_this_network(self):
    tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n        '
    t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
    networks = server.properties['networks']
    for network in networks:
        server._validate_network(network)
    self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='not_this_network')
    ex = self.assertRaises(exception.StackValidationFailed, server._build_nics, networks)
    self.assertEqual('Specified subnet 1234 does not belongs to network 4321.', str(ex))