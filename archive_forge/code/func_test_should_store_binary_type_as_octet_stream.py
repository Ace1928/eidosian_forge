import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_store_binary_type_as_octet_stream(self):
    """We use bytes as the canonical binary type.

        The client should base64 encode the payload before sending the
        request.
        """
    data = {'secret_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    binary_payload = b'F\x130\x89f\x8e\xd9\xa1\x0e\x1f\r\xf67uu\x8b'
    secret = self.manager.create()
    secret.name = self.secret.name
    secret.payload = binary_payload
    secret.store()
    secret_req = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(self.secret.name, secret_req['name'])
    self.assertEqual('application/octet-stream', secret_req['payload_content_type'])
    self.assertEqual('base64', secret_req['payload_content_encoding'])
    self.assertNotEqual(binary_payload, secret_req['payload'])