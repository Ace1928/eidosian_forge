import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
class UserPasswordHashingTestsNoCompat(test_backend_sql.SqlTests):

    def config_overrides(self):
        super(UserPasswordHashingTestsNoCompat, self).config_overrides()
        self.config_fixture.config(group='identity', password_hash_algorithm='scrypt')

    def test_configured_algorithm_used(self):
        with sql.session_for_read() as session:
            user_ref = PROVIDERS.identity_api._get_user(session, self.user_foo['id'])
        self.assertEqual(passlib.hash.scrypt, password_hashing._get_hasher_from_ident(user_ref.password))