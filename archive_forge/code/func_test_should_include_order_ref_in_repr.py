from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_include_order_ref_in_repr(self):
    order_args = self._get_order_args(self.key_order_data)
    order_obj = orders.KeyOrder(api=None, **order_args)
    self.assertIn('order_ref=' + self.entity_href, repr(order_obj))