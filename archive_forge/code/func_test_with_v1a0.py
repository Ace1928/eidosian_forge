from oslo_utils import importutils
from blazarclient import client
from blazarclient import exception
from blazarclient import tests
def test_with_v1a0(self):
    self.client.Client(version='1a0')
    self.import_obj.assert_called_once_with('blazarclient.v1.client.Client', service_type='reservation')