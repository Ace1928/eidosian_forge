import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def test_normalize_network(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'networks': [{'name': 'private'}]}
    result = c._fix_backwards_networks(cloud)
    expected = {'networks': [{'name': 'private', 'routes_externally': False, 'nat_destination': False, 'default_interface': False, 'nat_source': False, 'routes_ipv4_externally': False, 'routes_ipv6_externally': False}]}
    self.assertEqual(expected, result)