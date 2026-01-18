from oslo_utils import importutils
from blazarclient import client
from blazarclient import exception
from blazarclient import tests
def test_with_v1(self):
    self.client.Client()
    self.import_obj.assert_called_once_with('blazarclient.v1.client.Client', service_type='reservation')