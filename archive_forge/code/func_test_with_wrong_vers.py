from oslo_utils import importutils
from blazarclient import client
from blazarclient import exception
from blazarclient import tests
def test_with_wrong_vers(self):
    self.assertRaises(exception.UnsupportedVersion, self.client.Client, version='0.0')