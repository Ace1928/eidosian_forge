import os
import uuid
from keystone.common import jwt_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.token import provider
from keystone.token.providers import jws
class TestJWSProvider(unit.TestCase):

    def setUp(self):
        super(TestJWSProvider, self).setUp()
        self.config_fixture.config(group='token', provider='jws')
        self.useFixture(ksfixtures.JWSKeyRepository(self.config_fixture))
        self.provider = jws.Provider()

    def test_invalid_token_raises_token_not_found(self):
        token_id = uuid.uuid4().hex
        self.assertRaises(exception.TokenNotFound, self.provider.validate_token, token_id)

    def test_non_existent_private_key_raises_system_exception(self):
        private_key = os.path.join(CONF.jwt_tokens.jws_private_key_repository, 'private.pem')
        os.remove(private_key)
        self.assertRaises(SystemExit, jws.Provider)

    def test_non_existent_public_key_repo_raises_system_exception(self):
        for f in os.listdir(CONF.jwt_tokens.jws_public_key_repository):
            path = os.path.join(CONF.jwt_tokens.jws_public_key_repository, f)
            os.remove(path)
        os.rmdir(CONF.jwt_tokens.jws_public_key_repository)
        self.assertRaises(SystemExit, jws.Provider)

    def test_empty_public_key_repo_raises_system_exception(self):
        for f in os.listdir(CONF.jwt_tokens.jws_public_key_repository):
            path = os.path.join(CONF.jwt_tokens.jws_public_key_repository, f)
            os.remove(path)
        self.assertRaises(SystemExit, jws.Provider)

    def test_unable_to_verify_token_with_missing_public_key(self):
        token = token_model.TokenModel()
        token.methods = ['password']
        token.user_id = uuid.uuid4().hex
        token.audit_id = provider.random_urlsafe_str()
        token.expires_at = utils.isotime(provider.default_expire_time(), subsecond=True)
        token_id, issued_at = self.provider.generate_id_and_issued_at(token)
        current_pub_key = os.path.join(CONF.jwt_tokens.jws_public_key_repository, 'public.pem')
        os.remove(current_pub_key)
        for _ in range(2):
            private_key_path = os.path.join(CONF.jwt_tokens.jws_private_key_repository, uuid.uuid4().hex)
            pub_key_path = os.path.join(CONF.jwt_tokens.jws_public_key_repository, uuid.uuid4().hex)
            jwt_utils.create_jws_keypair(private_key_path, pub_key_path)
        self.assertRaises(exception.TokenNotFound, self.provider.validate_token, token_id)

    def test_verify_token_with_multiple_public_keys_present(self):
        token = token_model.TokenModel()
        token.methods = ['password']
        token.user_id = uuid.uuid4().hex
        token.audit_id = provider.random_urlsafe_str()
        token.expires_at = utils.isotime(provider.default_expire_time(), subsecond=True)
        token_id, issued_at = self.provider.generate_id_and_issued_at(token)
        for _ in range(2):
            private_key_path = os.path.join(CONF.jwt_tokens.jws_private_key_repository, uuid.uuid4().hex)
            pub_key_path = os.path.join(CONF.jwt_tokens.jws_public_key_repository, uuid.uuid4().hex)
            jwt_utils.create_jws_keypair(private_key_path, pub_key_path)
        self.provider.validate_token(token_id)