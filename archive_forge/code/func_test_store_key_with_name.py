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
def test_store_key_with_name(self):
    secret_key = bytes(b'\x01\x02\xa0\xb3')
    key_length = len(secret_key) * 8
    secret_name = 'My Secret'
    _key = sym_key.SymmetricKey('AES', key_length, secret_key, secret_name)
    secret = mock.Mock()
    self.create.return_value = secret
    secret.store.return_value = self.secret_ref
    returned_uuid = self.key_mgr.store(self.ctxt, _key)
    self.create.assert_called_once_with(algorithm='AES', bit_length=key_length, payload=secret_key, name=secret_name, secret_type='symmetric')
    self.assertEqual(self.key_id, returned_uuid)