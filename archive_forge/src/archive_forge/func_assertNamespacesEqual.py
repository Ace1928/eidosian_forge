from oslo_utils.fixture import uuidsentinel as uuids
import requests
from glance.tests import functional
def assertNamespacesEqual(self, actual, expected):
    """Assert two namespace dictionaries are the same."""
    actual.pop('created_at', None)
    actual.pop('updated_at', None)
    expected_namespace = {'namespace': expected['namespace'], 'display_name': expected['display_name'], 'description': expected['description'], 'visibility': expected['visibility'], 'protected': False, 'owner': expected['owner'], 'self': '/v2/metadefs/namespaces/%s' % expected['namespace'], 'schema': '/v2/schemas/metadefs/namespace'}
    self.assertEqual(actual, expected_namespace)