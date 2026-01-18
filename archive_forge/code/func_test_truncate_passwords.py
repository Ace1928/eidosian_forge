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
def test_truncate_passwords(self):
    user = self._create_user(uuid.uuid4().hex)
    self._add_passwords_to_history(user, n=4)
    user_ref = self._get_user_ref(user['id'])
    self.assertEqual(len(user_ref.local_user.passwords), self.max_cnt + 1)