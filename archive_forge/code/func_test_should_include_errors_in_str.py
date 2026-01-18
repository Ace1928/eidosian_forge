from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_include_errors_in_str(self):
    order_args = self._get_order_args(self.key_order_data)
    error_code = 500
    error_reason = 'Something is broken'
    order_obj = orders.KeyOrder(api=None, error_status_code=error_code, error_reason=error_reason, **order_args)
    self.assertIn(str(error_code), str(order_obj))
    self.assertIn(error_reason, str(order_obj))