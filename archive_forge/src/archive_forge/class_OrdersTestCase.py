from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
class OrdersTestCase(test_client.BaseEntityResource):

    def setUp(self):
        self._setUp('orders', entity_id='d0460cc4-2876-4493-b7de-fc5c812883cc')
        self.secret_ref = self.endpoint + '/secrets/a2292306-6da0-4f60-bd8a-84fc8d692716'
        self.key_order_data = '{{\n            "status": "ACTIVE",\n            "secret_ref": "{0}",\n            "updated": "2014-10-21T17:15:50.871596",\n            "meta": {{\n                "name": "secretname",\n                "algorithm": "aes",\n                "payload_content_type": "application/octet-stream",\n                "mode": "cbc",\n                "bit_length": 256,\n                "expiration": "2015-02-28T19:14:44.180394"\n            }},\n            "created": "2014-10-21T17:15:50.824202",\n            "type": "key",\n            "order_ref": "{1}"\n        }}'.format(self.secret_ref, self.entity_href)
        self.key_order_invalid_data = '{{\n            "status": "ACTIVE",\n            "secret_ref": "{0}",\n            "updated": "2014-10-21T17:15:50.871596",\n            "meta": {{\n                "name": "secretname",\n                "algorithm": "aes",\n                "request_type":"invalid",\n                "payload_content_type": "application/octet-stream",\n                "mode": "cbc",\n                "bit_length": 256,\n                "expiration": "2015-02-28T19:14:44.180394"\n            }},\n            "created": "2014-10-21T17:15:50.824202",\n            "type": "key",\n            "order_ref": "{1}"\n        }}'.format(self.secret_ref, self.entity_href)
        self.container_ref = self.endpoint + '/containers/a2292306-6da0-4f60-bd8a-84fc8d692716'
        self.source_container_ref = self.endpoint + '/containers/c6f20480-c1e5-442b-94a0-cb3b5e0cf179'
        self.cert_order_data = '{{\n            "status": "ACTIVE",\n            "container_ref": "{0}",\n            "updated": "2014-10-21T17:15:50.871596",\n            "meta": {{\n                "name": "secretname",\n                "subject_dn": "cn=server.example.com,o=example.com",\n                "request_type": "stored-key",\n                "container_ref": "{1}"\n            }},\n            "created": "2014-10-21T17:15:50.824202",\n            "type": "certificate",\n            "order_ref": "{2}"\n        }}'.format(self.container_ref, self.source_container_ref, self.entity_href)
        self.manager = self.client.orders

    def _get_order_args(self, order_data):
        order_args = jsonutils.loads(order_data)
        order_args.update(order_args.pop('meta'))
        order_args.pop('type')
        return order_args