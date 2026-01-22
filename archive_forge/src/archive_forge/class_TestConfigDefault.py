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
class TestConfigDefault(base.TestCase):

    def setUp(self):
        super(TestConfigDefault, self).setUp()
        self.addCleanup(self._reset_defaults)

    def _reset_defaults(self):
        defaults._defaults = None

    def test_set_no_default(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
        cc = c.get_one(cloud='_test-cloud_', argparse=None)
        self._assert_cloud_details(cc)
        self.assertEqual('password', cc.auth_type)