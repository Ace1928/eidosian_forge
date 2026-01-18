import fixtures
import hashlib
import uuid
from oslo_log import log
from keystone.common import fernet_utils
from keystone.credential.providers import fernet as credential_fernet
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def test_warning_is_logged_when_encrypting_with_null_key(self):
    blob = uuid.uuid4().hex
    logging_fixture = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
    expected_output = 'Encrypting credentials with the null key. Please properly encrypt credentials using `keystone-manage credential_setup`, `keystone-manage credential_migrate`, and `keystone-manage credential_rotate`'
    encrypted_blob, primary_key_hash = self.provider.encrypt(blob)
    self.assertIn(expected_output, logging_fixture.output)