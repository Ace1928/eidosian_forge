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
def test_single_default_interface(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'networks': [{'name': 'blue', 'default_interface': True}, {'name': 'purple', 'default_interface': True}]}
    self.assertRaises(exceptions.ConfigException, c._fix_backwards_networks, cloud)