from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_delete(self, order_ref=None):
    order_ref = order_ref or self.entity_href
    self.responses.delete(self.entity_href, status_code=204)
    self.manager.delete(order_ref=order_ref)
    self.assertEqual(self.entity_href, self.responses.last_request.url)