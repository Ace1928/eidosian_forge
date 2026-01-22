from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
class WhenTestingKeyOrders(OrdersTestCase):

    def test_should_include_errors_in_str(self):
        order_args = self._get_order_args(self.key_order_data)
        error_code = 500
        error_reason = 'Something is broken'
        order_obj = orders.KeyOrder(api=None, error_status_code=error_code, error_reason=error_reason, **order_args)
        self.assertIn(str(error_code), str(order_obj))
        self.assertIn(error_reason, str(order_obj))

    def test_should_include_order_ref_in_repr(self):
        order_args = self._get_order_args(self.key_order_data)
        order_obj = orders.KeyOrder(api=None, **order_args)
        self.assertIn('order_ref=' + self.entity_href, repr(order_obj))

    def test_should_be_immutable_after_submit(self):
        data = {'order_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        order = self.manager.create_key(name='name', algorithm='algorithm', payload_content_type='payload_content_type')
        order_href = order.submit()
        self.assertEqual(self.entity_href, order_href)
        attributes = ['name', 'expiration', 'algorithm', 'bit_length', 'mode', 'payload_content_type']
        for attr in attributes:
            try:
                setattr(order, attr, 'test')
                self.fail("didn't raise an ImmutableException exception")
            except base.ImmutableException:
                pass

    def test_should_submit_via_constructor(self):
        data = {'order_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        order = self.manager.create_key(name='name', algorithm='algorithm', payload_content_type='payload_content_type')
        order_href = order.submit()
        self.assertEqual(self.entity_href, order_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        order_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual('name', order_req['meta']['name'])
        self.assertEqual('algorithm', order_req['meta']['algorithm'])
        self.assertEqual('payload_content_type', order_req['meta']['payload_content_type'])

    def test_should_submit_via_attributes(self):
        data = {'order_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        order = self.manager.create_key()
        order.name = 'name'
        order.algorithm = 'algorithm'
        order.payload_content_type = 'payload_content_type'
        order_href = order.submit()
        self.assertEqual(self.entity_href, order_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        order_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual('name', order_req['meta']['name'])
        self.assertEqual('algorithm', order_req['meta']['algorithm'])
        self.assertEqual('payload_content_type', order_req['meta']['payload_content_type'])

    def test_should_not_be_able_to_set_generated_attributes(self):
        order = self.manager.create_key()
        attributes = ['order_ref', 'secret_ref', 'created', 'updated', 'status', 'error_status_code', 'error_reason']
        for attr in attributes:
            try:
                setattr(order, attr, 'test')
                self.fail("didn't raise an AttributeError exception")
            except AttributeError:
                pass

    def test_should_delete_from_object(self, order_ref=None):
        order_ref = order_ref or self.entity_href
        data = {'order_ref': order_ref}
        self.responses.post(self.entity_base + '/', json=data)
        self.responses.delete(self.entity_href, status_code=204)
        order = self.manager.create_key(name='name', algorithm='algorithm', payload_content_type='payload_content_type')
        order_href = order.submit()
        self.assertEqual(order_ref, order_href)
        order.delete()
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_delete_from_object_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_delete_from_object(bad_href)

    def test_should_delete_from_object_using_only_uuid(self):
        self.test_should_delete_from_object(self.entity_id)