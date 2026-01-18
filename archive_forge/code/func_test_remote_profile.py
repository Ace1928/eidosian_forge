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
def test_remote_profile(self):
    single_conf = base._write_yaml({'clouds': {'remote': {'profile': 'https://example.com', 'auth': {'username': 'testuser', 'password': 'testpass', 'project_name': 'testproject'}, 'region_name': 'test-region'}}})
    self.register_uris([dict(method='GET', uri='https://example.com/.well-known/openstack/api', json={'name': 'example', 'profile': {'auth': {'auth_url': 'https://auth.example.com/v3'}}})])
    c = config.OpenStackConfig(config_files=[single_conf])
    cc = c.get_one(cloud='remote')
    self.assertEqual(cc.name, 'remote')
    self.assertEqual(cc.auth['auth_url'], 'https://auth.example.com/v3')
    self.assertEqual(cc.auth['username'], 'testuser')