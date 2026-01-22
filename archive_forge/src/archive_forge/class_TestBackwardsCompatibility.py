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
class TestBackwardsCompatibility(base.TestCase):

    def test_set_no_default(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'identity_endpoint_type': 'admin', 'compute_endpoint_type': 'private', 'endpoint_type': 'public', 'auth_type': 'v3password'}
        result = c._fix_backwards_interface(cloud)
        expected = {'identity_interface': 'admin', 'compute_interface': 'private', 'interface': 'public', 'auth_type': 'v3password'}
        self.assertDictEqual(expected, result)

    def test_project_v2password(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'auth_type': 'v2password', 'auth': {'project-name': 'my_project_name', 'project-id': 'my_project_id'}}
        result = c._fix_backwards_project(cloud)
        expected = {'auth_type': 'v2password', 'auth': {'tenant_name': 'my_project_name', 'tenant_id': 'my_project_id'}}
        self.assertEqual(expected, result)

    def test_project_password(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'auth_type': 'password', 'auth': {'project-name': 'my_project_name', 'project-id': 'my_project_id'}}
        result = c._fix_backwards_project(cloud)
        expected = {'auth_type': 'password', 'auth': {'project_name': 'my_project_name', 'project_id': 'my_project_id'}}
        self.assertEqual(expected, result)

    def test_project_conflict_priority(self):
        """The order of priority should be
        1: env or cli settings
        2: setting from 'auth' section of clouds.yaml

        The ordering of #1 is important so that operators can use domain-wide
        inherited credentials in clouds.yaml.
        """
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'auth_type': 'password', 'auth': {'project_id': 'my_project_id'}}
        result = c._fix_backwards_project(cloud)
        expected = {'auth_type': 'password', 'auth': {'project_id': 'my_project_id'}}
        self.assertEqual(expected, result)
        cloud = {'auth_type': 'password', 'auth': {'project_id': 'my_project_id'}, 'project_id': 'different_project_id'}
        result = c._fix_backwards_project(cloud)
        expected = {'auth_type': 'password', 'auth': {'project_id': 'different_project_id'}}
        self.assertEqual(expected, result)

    def test_backwards_network_fail(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'external_network': 'public', 'networks': [{'name': 'private', 'routes_externally': False}]}
        self.assertRaises(exceptions.ConfigException, c._fix_backwards_networks, cloud)

    def test_backwards_network(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'external_network': 'public', 'internal_network': 'private'}
        result = c._fix_backwards_networks(cloud)
        expected = {'external_network': 'public', 'internal_network': 'private', 'networks': [{'name': 'public', 'routes_externally': True, 'nat_destination': False, 'default_interface': True}, {'name': 'private', 'routes_externally': False, 'nat_destination': True, 'default_interface': False}]}
        self.assertEqual(expected, result)

    def test_normalize_network(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'networks': [{'name': 'private'}]}
        result = c._fix_backwards_networks(cloud)
        expected = {'networks': [{'name': 'private', 'routes_externally': False, 'nat_destination': False, 'default_interface': False, 'nat_source': False, 'routes_ipv4_externally': False, 'routes_ipv6_externally': False}]}
        self.assertEqual(expected, result)

    def test_single_default_interface(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cloud = {'networks': [{'name': 'blue', 'default_interface': True}, {'name': 'purple', 'default_interface': True}]}
        self.assertRaises(exceptions.ConfigException, c._fix_backwards_networks, cloud)