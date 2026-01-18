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
def test_validate_password_history_with_invalid_password(self):
    password = uuid.uuid4().hex
    user = self._create_user(password)
    with self.make_request():
        self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user['id'], original_password=password, new_password=password)
        new_password = uuid.uuid4().hex
        self.assertValidChangePassword(user['id'], password, new_password)
        self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user['id'], original_password=new_password, new_password=password)