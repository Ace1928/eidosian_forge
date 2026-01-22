from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
class WhenTestingOrderManager(OrdersTestCase):

    def test_should_get(self, order_ref=None):
        order_ref = order_ref or self.entity_href
        self.responses.get(self.entity_href, text=self.key_order_data)
        order = self.manager.get(order_ref=order_ref)
        self.assertIsInstance(order, orders.KeyOrder)
        self.assertEqual(self.entity_href, order.order_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_get_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_get(bad_href)

    def test_should_get_using_only_uuid(self):
        self.test_should_get(self.entity_id)

    def test_should_get_invalid_meta(self):
        self.responses.get(self.entity_href, text=self.key_order_invalid_data)
        self.assertRaises(TypeError, self.manager.get, self.entity_href)

    def test_should_get_list(self):
        data = {'orders': [jsonutils.loads(self.key_order_data) for _ in range(3)]}
        self.responses.get(self.entity_base, json=data)
        orders_list = self.manager.list(limit=10, offset=5)
        self.assertTrue(len(orders_list) == 3)
        self.assertIsInstance(orders_list[0], orders.KeyOrder)
        self.assertEqual(self.entity_href, orders_list[0].order_ref)
        self.assertEqual(self.entity_base, self.responses.last_request.url.split('?')[0])
        self.assertEqual(['10'], self.responses.last_request.qs['limit'])
        self.assertEqual(['5'], self.responses.last_request.qs['offset'])

    def test_should_delete(self, order_ref=None):
        order_ref = order_ref or self.entity_href
        self.responses.delete(self.entity_href, status_code=204)
        self.manager.delete(order_ref=order_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_delete_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_delete(bad_href)

    def test_should_delete_using_only_uuid(self):
        self.test_should_delete(self.entity_id)

    def test_should_fail_delete_no_href(self):
        self.assertRaises(ValueError, self.manager.delete, None)

    def test_should_get_total(self):
        self.responses.get(self.entity_base, json={'total': 1})
        total = self.manager.total()
        self.assertEqual(1, total)

    def test_get_formatted_data(self):
        self.responses.get(self.entity_href, text=self.key_order_data)
        order = self.manager.get(order_ref=self.entity_href)
        data = order._get_formatted_data()
        order_args = self._get_order_args(self.key_order_data)
        self.assertEqual(timeutils.parse_isotime(order_args['created']).isoformat(), data[4])