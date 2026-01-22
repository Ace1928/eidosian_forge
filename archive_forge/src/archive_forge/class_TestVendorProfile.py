import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
class TestVendorProfile(base.TestCase):

    def setUp(self):
        super(TestVendorProfile, self).setUp()
        config_dir = self.useFixture(fixtures.TempDir()).path
        config_path = os.path.join(config_dir, 'clouds.yaml')
        public_clouds = os.path.join(config_dir, 'clouds-public.yaml')
        with open(config_path, 'w') as conf:
            conf.write(CLOUD_CONFIG)
        with open(public_clouds, 'w') as conf:
            conf.write(PUBLIC_CLOUDS_YAML)
        self.useFixture(fixtures.EnvironmentVariable('OS_CLIENT_CONFIG_FILE', config_path))
        self.use_keystone_v2()
        self.config = openstack.config.loader.OpenStackConfig(vendor_files=[public_clouds])

    def test_conn_from_profile(self):
        self.cloud = self.config.get_one(cloud='profiled-cloud')
        conn = connection.Connection(config=self.cloud)
        self.assertIsNotNone(conn)

    def test_hook_from_profile(self):
        self.cloud = self.config.get_one(cloud='profiled-cloud')
        conn = connection.Connection(config=self.cloud)
        self.assertEqual('test_val', conn.test)

    def test_hook_from_connection_param(self):
        conn = connection.Connection(cloud='sample-cloud', vendor_hook='openstack.tests.unit.test_connection:vendor_hook')
        self.assertEqual('test_val', conn.test)

    def test_hook_from_connection_ignore_missing(self):
        conn = connection.Connection(cloud='sample-cloud', vendor_hook='openstack.tests.unit.test_connection:missing')
        self.assertIsNotNone(conn)