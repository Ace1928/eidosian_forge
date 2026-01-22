from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def _list_consumers(self, secret_ref, consumers=[]):
    mock_get_secret_for_client(self.client, consumers)
    return self.manager.list_consumers(secret_ref)