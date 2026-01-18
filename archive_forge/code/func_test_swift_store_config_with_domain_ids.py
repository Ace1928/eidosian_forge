import fixtures
from glance_store._drivers.swift import utils as sutils
from glance_store import exceptions
from glance_store.tests import base
def test_swift_store_config_with_domain_ids(self):
    swift_params = sutils.SwiftParams(self.conf).params
    self.assertEqual('projdomainid', swift_params['ref4']['project_domain_id'])
    self.assertIsNone(swift_params['ref4']['project_domain_name'])
    self.assertEqual('userdomainid', swift_params['ref4']['user_domain_id'])
    self.assertIsNone(swift_params['ref4']['user_domain_name'])