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
class MinimumPasswordAgeTests(test_backend_sql.SqlTests):

    def setUp(self):
        super(MinimumPasswordAgeTests, self).setUp()
        self.config_fixture.config(group='security_compliance', minimum_password_age=1)
        self.initial_password = uuid.uuid4().hex
        self.user = self._create_new_user(self.initial_password)

    def test_user_cannot_change_password_before_min_age(self):
        new_password = uuid.uuid4().hex
        self.assertValidChangePassword(self.user['id'], self.initial_password, new_password)
        with self.make_request():
            self.assertRaises(exception.PasswordAgeValidationError, PROVIDERS.identity_api.change_password, user_id=self.user['id'], original_password=new_password, new_password=uuid.uuid4().hex)

    def test_user_can_change_password_after_min_age(self):
        new_password = uuid.uuid4().hex
        self.assertValidChangePassword(self.user['id'], self.initial_password, new_password)
        password_created_at = datetime.datetime.utcnow() - datetime.timedelta(days=CONF.security_compliance.minimum_password_age + 1)
        self._update_password_created_at(self.user['id'], password_created_at)
        self.assertValidChangePassword(self.user['id'], new_password, uuid.uuid4().hex)

    def test_user_can_change_password_after_admin_reset(self):
        new_password = uuid.uuid4().hex
        self.assertValidChangePassword(self.user['id'], self.initial_password, new_password)
        with self.make_request():
            self.assertRaises(exception.PasswordAgeValidationError, PROVIDERS.identity_api.change_password, user_id=self.user['id'], original_password=new_password, new_password=uuid.uuid4().hex)
        new_password = uuid.uuid4().hex
        self.user['password'] = new_password
        PROVIDERS.identity_api.update_user(self.user['id'], self.user)
        self.assertValidChangePassword(self.user['id'], new_password, uuid.uuid4().hex)

    def assertValidChangePassword(self, user_id, password, new_password):
        with self.make_request():
            PROVIDERS.identity_api.change_password(user_id=user_id, original_password=password, new_password=new_password)
            PROVIDERS.identity_api.authenticate(user_id=user_id, password=new_password)

    def _create_new_user(self, password):
        user = {'name': uuid.uuid4().hex, 'domain_id': CONF.identity.default_domain_id, 'enabled': True, 'password': password}
        return PROVIDERS.identity_api.create_user(user)

    def _update_password_created_at(self, user_id, password_create_at):
        with sql.session_for_write() as session:
            user_ref = session.get(model.User, user_id)
            latest_password = user_ref.password_ref
            slightly_less = datetime.timedelta(minutes=1)
            for password_ref in user_ref.local_user.passwords:
                password_ref.created_at = password_create_at - slightly_less
            latest_password.created_at = password_create_at