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
@mock.patch('openstack.config.cloud_region.keyring')
def test_set_auth_cache_empty_auth(self, kr_mock):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], secure_files=[])
    c._cache_auth = True
    kr_mock.get_password = mock.Mock(side_effect=[RuntimeError])
    kr_mock.set_password = mock.Mock()
    region = c.get_one('_test-cloud_')
    region.set_auth_cache()
    kr_mock.set_password.assert_not_called()