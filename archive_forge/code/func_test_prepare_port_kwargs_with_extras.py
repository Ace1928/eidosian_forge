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
def test_prepare_port_kwargs_with_extras(self):
    tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                  port_extra_properties:\n                    mac_address: 00:00:00:00:00:00\n                    allowed_address_pairs:\n                      - ip_address: 127.0.0.1\n                        mac_address: None\n                      - mac_address: 00:00:00:00:00:00\n\n        '
    t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
    network = {'network': '4321', 'subnet': '1234', 'fixed_ip': '127.0.0.1', 'port_extra_properties': {'value_specs': {}, 'mac_address': '00:00:00:00:00:00', 'allowed_address_pairs': [{'ip_address': '127.0.0.1', 'mac_address': None}, {'mac_address': '00:00:00:00:00:00'}]}}
    sec_uuids = ['8d94c72093284da88caaef5e985d96f7']
    self.patchobject(neutron.NeutronClientPlugin, 'get_secgroup_uuids', return_value=sec_uuids)
    kwargs = server._prepare_internal_port_kwargs(network, security_groups=['test_sec'])
    self.assertEqual({'network_id': '4321', 'security_groups': sec_uuids, 'fixed_ips': [{'ip_address': '127.0.0.1', 'subnet_id': '1234'}], 'mac_address': '00:00:00:00:00:00', 'allowed_address_pairs': [{'ip_address': '127.0.0.1'}, {'mac_address': '00:00:00:00:00:00'}]}, kwargs)