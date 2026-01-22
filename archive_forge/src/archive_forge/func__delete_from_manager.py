from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def _delete_from_manager(self, secret_ref, force=False, consumers=[]):
    mock_get_secret_for_client(self.client, consumers=consumers)
    mock_delete_secret_for_responses(self.responses, self.entity_href)
    self.manager.delete(secret_ref=secret_ref, force=force)