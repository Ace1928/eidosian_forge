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
class TestConfigPrompt(base.TestCase):

    def setUp(self):
        super(TestConfigPrompt, self).setUp()
        self.args = dict(auth_url='http://example.com/v2', username='user', project_name='project', auth_type='password')
        self.options = argparse.Namespace(**self.args)

    def test_get_one_prompt(self):
        c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], pw_func=prompt_for_password)
        cc = c.get_one(cloud='_test_cloud_no_vendor', argparse=self.options)
        self.assertEqual('promptpass', cc.auth['password'])