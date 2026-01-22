from oslo_utils import importutils
from blazarclient import client
from blazarclient import exception
from blazarclient import tests
class BaseClientTestCase(tests.TestCase):

    def setUp(self):
        super(BaseClientTestCase, self).setUp()
        self.client = client
        self.import_obj = self.patch(importutils, 'import_object')

    def test_with_v1(self):
        self.client.Client()
        self.import_obj.assert_called_once_with('blazarclient.v1.client.Client', service_type='reservation')

    def test_with_v1a0(self):
        self.client.Client(version='1a0')
        self.import_obj.assert_called_once_with('blazarclient.v1.client.Client', service_type='reservation')

    def test_with_wrong_vers(self):
        self.assertRaises(exception.UnsupportedVersion, self.client.Client, version='0.0')