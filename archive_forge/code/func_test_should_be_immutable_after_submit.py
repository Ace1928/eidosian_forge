from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def test_should_be_immutable_after_submit(self):
    data = {'order_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    order = self.manager.create_asymmetric(name='name', algorithm='algorithm', payload_content_type='payload_content_type')
    order_href = order.submit()
    self.assertEqual(self.entity_href, order_href)
    attributes = ['name', 'expiration', 'algorithm', 'bit_length', 'pass_phrase', 'payload_content_type']
    for attr in attributes:
        try:
            setattr(order, attr, 'test')
            self.fail("{0} didn't raise an ImmutableException exception".format(attr))
        except base.ImmutableException:
            pass