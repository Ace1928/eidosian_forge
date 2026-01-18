from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
def test_scheme_validation(self):
    valid_schemas = ('file://', 'http://')
    correct_uri = 'file://test'
    location.StoreLocation.validate_schemas(correct_uri, valid_schemas)
    incorrect_uri = 'fake://test'
    self.assertRaises(exceptions.BadStoreUri, location.StoreLocation.validate_schemas, incorrect_uri, valid_schemas)