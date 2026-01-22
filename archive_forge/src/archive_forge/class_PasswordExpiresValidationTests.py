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
class PasswordExpiresValidationTests(test_backend_sql.SqlTests):

    def setUp(self):
        super(PasswordExpiresValidationTests, self).setUp()
        self.password = uuid.uuid4().hex
        self.user_dict = self._get_test_user_dict(self.password)
        self.config_fixture.config(group='security_compliance', password_expires_days=90)

    def test_authenticate_with_expired_password(self):
        password_created_at = datetime.datetime.utcnow() - datetime.timedelta(days=CONF.security_compliance.password_expires_days + 1)
        user = self._create_user(self.user_dict, password_created_at)
        with self.make_request():
            self.assertRaises(exception.PasswordExpired, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=self.password)

    def test_authenticate_with_non_expired_password(self):
        password_created_at = datetime.datetime.utcnow() - datetime.timedelta(days=CONF.security_compliance.password_expires_days - 1)
        user = self._create_user(self.user_dict, password_created_at)
        with self.make_request():
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)

    def test_authenticate_with_expired_password_for_ignore_user_option(self):
        self.user_dict.setdefault('options', {})[iro.IGNORE_PASSWORD_EXPIRY_OPT.option_name] = False
        password_created_at = datetime.datetime.utcnow() - datetime.timedelta(days=CONF.security_compliance.password_expires_days + 1)
        user = self._create_user(self.user_dict, password_created_at)
        with self.make_request():
            self.assertRaises(exception.PasswordExpired, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=self.password)
            user['options'][iro.IGNORE_PASSWORD_EXPIRY_OPT.option_name] = True
            user = PROVIDERS.identity_api.update_user(user['id'], user)
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)

    def _get_test_user_dict(self, password):
        test_user_dict = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'domain_id': CONF.identity.default_domain_id, 'enabled': True, 'password': password}
        return test_user_dict

    def _create_user(self, user_dict, password_created_at):
        driver = PROVIDERS.identity_api.driver
        driver.create_user(user_dict['id'], user_dict)
        with sql.session_for_write() as session:
            user_ref = session.get(model.User, user_dict['id'])
            user_ref.password_ref.created_at = password_created_at
            user_ref.password_ref.expires_at = user_ref._get_password_expires_at(password_created_at)
            return base.filter_user(user_ref.to_dict())