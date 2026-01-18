from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
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