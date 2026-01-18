import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_decrypt(self, secret_ref=None):
    secret_ref = secret_ref or self.entity_href
    content_types_dict = {'default': 'application/octet-stream'}
    json = self.secret.get_dict(secret_ref, content_types_dict)
    metadata_response = self.responses.get(self.entity_href, request_headers={'Accept': 'application/json'}, json=json)
    decrypted = b'decrypted text here'
    request_headers = {'Accept': 'application/octet-stream'}
    decryption_response = self.responses.get(self.entity_payload_href, request_headers=request_headers, content=decrypted)
    secret = self.manager.get(secret_ref=secret_ref)
    secret_payload = secret.payload
    self.assertEqual(decrypted, secret_payload)
    self.assertEqual(self.entity_href, metadata_response.last_request.url)
    self.assertEqual(self.entity_payload_href, decryption_response.last_request.url)