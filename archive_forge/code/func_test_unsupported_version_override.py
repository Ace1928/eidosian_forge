import warnings
import testtools
from openstack import exceptions
from openstack import proxy
from openstack.tests.unit import base
def test_unsupported_version_override(self):
    self.cloud.config.config['image_api_version'] = '7'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.assertIsInstance(self.cloud.image, proxy.Proxy)
        self.assertEqual(1, len(w))
        self.assertIn('Service image has no discoverable version.', str(w[-1].message))
    self.assert_calls()