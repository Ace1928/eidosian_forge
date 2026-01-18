from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_get_invalid_meta(self):
    self.responses.get(self.entity_href, text=self.key_order_invalid_data)
    self.assertRaises(TypeError, self.manager.get, self.entity_href)