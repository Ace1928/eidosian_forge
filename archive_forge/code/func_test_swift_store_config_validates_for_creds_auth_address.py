import fixtures
from glance.common import exception
from glance.common import swift_store_utils
from glance.tests.unit import base
def test_swift_store_config_validates_for_creds_auth_address(self):
    swift_params = swift_store_utils.SwiftParams().params
    self.assertEqual('tenant:user1', swift_params['ref1']['user'])
    self.assertEqual('key1', swift_params['ref1']['key'])
    self.assertEqual('example.com', swift_params['ref1']['auth_address'])
    self.assertEqual('user2', swift_params['ref2']['user'])
    self.assertEqual('key2', swift_params['ref2']['key'])
    self.assertEqual('http://example.com', swift_params['ref2']['auth_address'])