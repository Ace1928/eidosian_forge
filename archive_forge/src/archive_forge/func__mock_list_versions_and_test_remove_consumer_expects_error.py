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
def _mock_list_versions_and_test_remove_consumer_expects_error(self, Error, ctxt, obj_ref, side_effect=None, service='storage', resource_type='volume', resource_id=uuidutils.generate_uuid()):
    self.mock_barbican.secrets.remove_consumer = mock.Mock(side_effect=side_effect)
    self._mock_list_versions_and_test_consumer_expects_error(Error, self.key_mgr.remove_consumer, ctxt, obj_ref, service=service, resource_type=resource_type, resource_id=resource_id)