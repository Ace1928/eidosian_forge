import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
@mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
def test_shell_args_cloud_no_vendor(self, config_mock):
    """Test cloud config options without the vendor file"""
    config_mock.return_value = ('file.yaml', copy.deepcopy(CLOUD_1))
    _shell = utils.make_shell()
    utils.fake_execute(_shell, '--os-cloud scc module list')
    self.assertEqual('scc', _shell.cloud.name)
    self.assertEqual(DEFAULT_AUTH_URL, _shell.cloud.config['auth']['auth_url'])
    self.assertEqual(DEFAULT_PROJECT_NAME, _shell.cloud.config['auth']['project_name'])
    self.assertEqual('zaphod', _shell.cloud.config['auth']['username'])
    self.assertEqual('occ-cloud', _shell.cloud.config['region_name'])
    self.assertEqual('occ-cloud', _shell.client_manager.region_name)
    self.assertEqual('glazed', _shell.cloud.config['donut'])
    self.assertEqual('admin', _shell.cloud.config['interface'])
    self.assertIsNone(_shell.cloud.config['cert'])
    self.assertIsNone(_shell.cloud.config['key'])
    self.assertIsNone(_shell.client_manager.cert)