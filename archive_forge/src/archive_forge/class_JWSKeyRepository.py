import os
import fixtures
from keystone.common import jwt_utils
from keystone.common import utils
class JWSKeyRepository(fixtures.Fixture):

    def __init__(self, config_fixture):
        super(JWSKeyRepository, self).__init__()
        self.config_fixture = config_fixture
        self.key_group = 'jwt_tokens'

    def setUp(self):
        super(JWSKeyRepository, self).setUp()
        private_key_directory = self.useFixture(fixtures.TempDir()).path
        public_key_directory = self.useFixture(fixtures.TempDir()).path
        self.config_fixture.config(group=self.key_group, jws_private_key_repository=private_key_directory)
        self.config_fixture.config(group=self.key_group, jws_public_key_repository=public_key_directory)
        utils.create_directory(private_key_directory)
        utils.create_directory(public_key_directory)
        private_key_path = os.path.join(private_key_directory, 'private.pem')
        public_key_path = os.path.join(public_key_directory, 'public.pem')
        jwt_utils.create_jws_keypair(private_key_path, public_key_path)