import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_store_with_deprecated_content_type(self):
    """DEPRECATION WARNING

        Manually setting the payload_content_type is deprecated and will be
        removed in a future release.
        """
    data = {'secret_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    payload = 'I should be octet-stream'
    payload_content_type = 'text/plain'
    secret = self.manager.create()
    secret.payload = payload
    secret.payload_content_type = payload_content_type
    secret.store()
    secret_req = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(payload, secret_req['payload'])
    self.assertEqual(payload_content_type, secret_req['payload_content_type'])