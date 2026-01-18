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
def test_validate_password_history_with_valid_password_only_once(self):
    self.config_fixture.config(group='security_compliance', unique_last_password_count=1)
    passwords = [uuid.uuid4().hex, uuid.uuid4().hex]
    user = self._create_user(passwords[0])
    self.assertValidChangePassword(user['id'], passwords[0], passwords[1])
    self.assertValidChangePassword(user['id'], passwords[1], passwords[0])