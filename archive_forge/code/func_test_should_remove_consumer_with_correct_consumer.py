from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def test_should_remove_consumer_with_correct_consumer(self):
    self._remove_consumer()
    self.assertEqual(self.consumers_delete_resource, self.responses.last_request.url)
    body = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(self.secret.consumer, body)