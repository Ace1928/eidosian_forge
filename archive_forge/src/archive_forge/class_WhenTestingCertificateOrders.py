from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
class WhenTestingCertificateOrders(OrdersTestCase):

    def test_get(self, order_ref=None):
        order_ref = order_ref or self.entity_href
        self.responses.get(self.entity_href, text=self.cert_order_data)
        order = self.manager.get(order_ref=order_ref)
        self.assertIsInstance(order, orders.CertificateOrder)
        self.assertEqual(self.entity_href, order.order_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_get_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_get(bad_href)

    def test_get_using_only_uuid(self):
        self.test_get(self.entity_id)

    def test_repr(self):
        order_args = self._get_order_args(self.cert_order_data)
        order_obj = orders.CertificateOrder(api=None, **order_args)
        self.assertIn('order_ref=' + self.entity_href, repr(order_obj))

    def test_constructor(self):
        data = {'order_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        order = self.manager.create_certificate(name='name', subject_dn='cn=server.example.com,o=example.com', request_type='stored-key', source_container_ref=self.source_container_ref)
        order_href = order.submit()
        self.assertEqual(self.entity_href, order_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        order_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual('name', order_req['meta']['name'])
        self.assertEqual('cn=server.example.com,o=example.com', order_req['meta']['subject_dn'])
        self.assertEqual('stored-key', order_req['meta']['request_type'])
        self.assertEqual(self.source_container_ref, order_req['meta']['container_ref'])

    def test_list(self):
        data = {'orders': [jsonutils.loads(self.cert_order_data) for _ in range(3)]}
        self.responses.get(self.entity_base, json=data)
        orders_list = self.manager.list(limit=10, offset=5)
        self.assertEqual(3, len(orders_list))
        self.assertIsInstance(orders_list[0], orders.CertificateOrder)
        self.assertEqual(self.entity_href, orders_list[0].order_ref)

    def test_get_formatted_data(self):
        self.responses.get(self.entity_href, text=self.cert_order_data)
        order = self.manager.get(order_ref=self.entity_href)
        data = order._get_formatted_data()
        order_args = self._get_order_args(self.cert_order_data)
        self.assertEqual(timeutils.parse_isotime(order_args['created']).isoformat(), data[4])