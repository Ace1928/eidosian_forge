import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_update_from_manager(self, secret_ref=None):
    text_payload = 'time for an ice cold 🍺'
    secret_ref = secret_ref or self.entity_href
    self.responses.put(self.entity_href, status_code=204)
    self.manager.update(secret_ref=secret_ref, payload=text_payload)
    self.assertEqual(self.entity_href, self.responses.last_request.url)