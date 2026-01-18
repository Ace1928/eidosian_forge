import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_store_text_type_as_text_plain(self):
    """We use unicode string as the canonical text type."""
    data = {'secret_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    text_payload = 'time for an ice cold 🍺'
    secret = self.manager.create()
    secret.payload = text_payload
    secret.store()
    secret_req = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(text_payload, secret_req['payload'])
    self.assertEqual('text/plain', secret_req['payload_content_type'])