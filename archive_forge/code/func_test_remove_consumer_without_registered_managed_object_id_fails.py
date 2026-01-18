import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_remove_consumer_without_registered_managed_object_id_fails(self):
    side_effect = barbican_exceptions.HTTPClientError('Not Found: Secret not found.', status_code=404)
    self._mock_list_versions_and_test_add_consumer_expects_error(exception.ManagedObjectNotFoundError, self.ctxt, self.secret_ref, side_effect=side_effect)