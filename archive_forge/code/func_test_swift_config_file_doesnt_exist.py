import fixtures
from glance.common import exception
from glance.common import swift_store_utils
from glance.tests.unit import base
def test_swift_config_file_doesnt_exist(self):
    self.config(swift_store_config_file='fake-file.conf')
    self.assertRaises(exception.InvalidSwiftStoreConfiguration, swift_store_utils.SwiftParams)