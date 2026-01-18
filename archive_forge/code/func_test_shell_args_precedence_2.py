import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
@mock.patch('openstack.config.loader.OpenStackConfig._load_vendor_file')
@mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
def test_shell_args_precedence_2(self, config_mock, vendor_mock):
    """Test command line overriding environment and occ"""
    config_mock.return_value = ('file.yaml', copy.deepcopy(CLOUD_2))
    vendor_mock.return_value = ('file.yaml', copy.deepcopy(PUBLIC_1))
    _shell = utils.make_shell()
    utils.fake_execute(_shell, '--os-region-name krikkit list user')
    self.assertEqual('megacloud', _shell.cloud.name)
    self.assertEqual(DEFAULT_AUTH_URL, _shell.cloud.config['auth']['auth_url'])
    self.assertEqual('cake', _shell.cloud.config['donut'])
    self.assertEqual('heart-o-gold', _shell.cloud.config['auth']['project_name'])
    self.assertEqual('zaphod', _shell.cloud.config['auth']['username'])
    self.assertEqual('krikkit', _shell.cloud.config['region_name'])
    self.assertEqual('krikkit', _shell.client_manager.region_name)