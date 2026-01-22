import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
from keystone.credential.providers import fernet as credential_provider
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.credential.backends import sql as credential_sql
from keystone import exception
class SqlCredential(SqlTests):

    def _create_credential_with_user_id(self, user_id=None):
        if not user_id:
            user_id = uuid.uuid4().hex
        credential = unit.new_credential_ref(user_id=user_id, extra=uuid.uuid4().hex, type=uuid.uuid4().hex)
        PROVIDERS.credential_api.create_credential(credential['id'], credential)
        return credential

    def _validate_credential_list(self, retrieved_credentials, expected_credentials):
        self.assertEqual(len(expected_credentials), len(retrieved_credentials))
        retrieved_ids = [c['id'] for c in retrieved_credentials]
        for cred in expected_credentials:
            self.assertIn(cred['id'], retrieved_ids)

    def setUp(self):
        super(SqlCredential, self).setUp()
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_provider.MAX_ACTIVE_KEYS))
        self.credentials = []
        self.user_credentials = []
        for _ in range(3):
            cred = self._create_credential_with_user_id()
            self.user_credentials.append(cred)
            self.credentials.append(cred)
        for _ in range(3):
            cred = self._create_credential_with_user_id(self.user_foo['id'])
            self.user_credentials.append(cred)
            self.credentials.append(cred)

    def test_backend_credential_sql_hints_none(self):
        credentials = PROVIDERS.credential_api.list_credentials(hints=None)
        self._validate_credential_list(credentials, self.user_credentials)

    def test_backend_credential_sql_no_hints(self):
        credentials = PROVIDERS.credential_api.list_credentials()
        self._validate_credential_list(credentials, self.user_credentials)

    def test_backend_credential_sql_encrypted_string(self):
        cred_dict = {'id': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'hash': uuid.uuid4().hex, 'encrypted_blob': b'randomdata'}
        ref = credential_sql.CredentialModel.from_dict(cred_dict)
        self.assertIsInstance(ref.encrypted_blob, str)

    def test_credential_limits(self):
        config_fixture_ = self.user = self.useFixture(config_fixture.Config())
        config_fixture_.config(group='credential', user_limit=4)
        self._create_credential_with_user_id(self.user_foo['id'])
        self.assertRaises(exception.CredentialLimitExceeded, self._create_credential_with_user_id, self.user_foo['id'])