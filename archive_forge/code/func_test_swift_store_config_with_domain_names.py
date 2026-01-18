import fixtures
from glance_store._drivers.swift import utils as sutils
from glance_store import exceptions
from glance_store.tests import base
def test_swift_store_config_with_domain_names(self):
    swift_params = sutils.SwiftParams(self.conf).params
    self.assertIsNone(swift_params['ref5']['project_domain_id'])
    self.assertEqual('projdomain', swift_params['ref5']['project_domain_name'])
    self.assertIsNone(swift_params['ref5']['user_domain_id'])
    self.assertEqual('userdomain', swift_params['ref5']['user_domain_name'])