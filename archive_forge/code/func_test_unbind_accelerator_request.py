import copy
import uuid
from openstack.tests.unit import base
def test_unbind_accelerator_request(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json={'accelerator_requests': [ARQ_DICT]}), dict(method='PATCH', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json=ARQ_DICT)])
    properties = [{'path': '/hostname', 'op': 'remove'}, {'path': '/instance_uuid', 'op': 'remove'}, {'path': '/device_rp_uuid', 'op': 'remove'}]
    self.assertTrue(self.cloud.unbind_accelerator_request(ARQ_DICT['uuid'], properties))
    self.assert_calls()