import copy
import os
import tempfile
import fixtures
import yaml
from openstack.config import cloud_region
from openstack.tests.unit import base
def _assert_cloud_details(self, cc):
    self.assertIsInstance(cc, cloud_region.CloudRegion)
    self.assertTrue(hasattr(cc, 'auth'))
    self.assertIsInstance(cc.auth, dict)
    self.assertIsNone(cc.cloud)
    self.assertIn('username', cc.auth)
    self.assertEqual('testuser', cc.auth['username'])
    self.assertEqual('testpass', cc.auth['password'])
    self.assertFalse(cc.config['image_api_use_tasks'])
    self.assertTrue('project_name' in cc.auth or 'project_id' in cc.auth)
    if 'project_name' in cc.auth:
        self.assertEqual('testproject', cc.auth['project_name'])
    elif 'project_id' in cc.auth:
        self.assertEqual('testproject', cc.auth['project_id'])
    self.assertEqual(cc.get_cache_expiration_time(), 1)
    self.assertEqual(cc.get_cache_resource_expiration('server'), 5.0)
    self.assertEqual(cc.get_cache_resource_expiration('image'), 7.0)