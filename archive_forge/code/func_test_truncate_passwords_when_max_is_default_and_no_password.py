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
def test_truncate_passwords_when_max_is_default_and_no_password(self):
    expected_length = 1
    self.max_cnt = 1
    self.config_fixture.config(group='security_compliance', unique_last_password_count=self.max_cnt)
    user = {'name': uuid.uuid4().hex, 'domain_id': 'default', 'enabled': True}
    user = PROVIDERS.identity_api.create_user(user)
    self._add_passwords_to_history(user, n=1)
    user_ref = self._get_user_ref(user['id'])
    self.assertEqual(len(user_ref.local_user.passwords), expected_length)