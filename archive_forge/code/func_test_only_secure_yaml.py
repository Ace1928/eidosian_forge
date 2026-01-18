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
def test_only_secure_yaml(self):
    c = config.OpenStackConfig(config_files=['nonexistent'], vendor_files=['nonexistent'], secure_files=[self.secure_yaml])
    cc = c.get_one(cloud='_test_cloud_no_vendor', validate=False)
    self.assertEqual('testpass', cc.auth['password'])