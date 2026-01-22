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
def _fail_auth_repeatedly(self, user_id):
    wrong_password = uuid.uuid4().hex
    for _ in range(CONF.security_compliance.lockout_failure_attempts):
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user_id, password=wrong_password)