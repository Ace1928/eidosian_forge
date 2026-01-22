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
class DisableInactiveUserTests(test_backend_sql.SqlTests):

    def setUp(self):
        super(DisableInactiveUserTests, self).setUp()
        self.password = uuid.uuid4().hex
        self.user_dict = self._get_user_dict(self.password)
        self.max_inactive_days = 90
        self.config_fixture.config(group='security_compliance', disable_user_account_days_inactive=self.max_inactive_days)

    def test_authenticate_user_disabled_due_to_inactivity(self):
        last_active_at = datetime.datetime.utcnow() - datetime.timedelta(days=self.max_inactive_days + 1)
        user = self._create_user(self.user_dict, last_active_at.date())
        with self.make_request():
            self.assertRaises(exception.UserDisabled, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=self.password)
            user = PROVIDERS.identity_api.get_user(user['id'])
            self.assertFalse(user['enabled'])
            user['enabled'] = True
            PROVIDERS.identity_api.update_user(user['id'], user)
            user = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)
            self.assertTrue(user['enabled'])

    def test_authenticate_user_not_disabled_due_to_inactivity(self):
        last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(days=self.max_inactive_days - 1)).date()
        user = self._create_user(self.user_dict, last_active_at)
        with self.make_request():
            user = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)
        self.assertTrue(user['enabled'])

    def test_get_user_disabled_due_to_inactivity(self):
        user = PROVIDERS.identity_api.create_user(self.user_dict)
        last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(self.max_inactive_days + 1)).date()
        self._update_user_last_active_at(user['id'], last_active_at)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertFalse(user['enabled'])
        user['enabled'] = True
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertTrue(user['enabled'])

    def test_get_user_not_disabled_due_to_inactivity(self):
        user = PROVIDERS.identity_api.create_user(self.user_dict)
        self.assertTrue(user['enabled'])
        last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(self.max_inactive_days - 1)).date()
        self._update_user_last_active_at(user['id'], last_active_at)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertTrue(user['enabled'])

    def test_enabled_after_create_update_user(self):
        self.config_fixture.config(group='security_compliance', disable_user_account_days_inactive=90)
        del self.user_dict['enabled']
        user = PROVIDERS.identity_api.create_user(self.user_dict)
        user_ref = self._get_user_ref(user['id'])
        self.assertTrue(user_ref.enabled)
        now = datetime.datetime.utcnow().date()
        self.assertGreaterEqual(now, user_ref.last_active_at)
        user['enabled'] = True
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = self._get_user_ref(user['id'])
        self.assertTrue(user_ref.enabled)
        user['enabled'] = False
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = self._get_user_ref(user['id'])
        self.assertFalse(user_ref.enabled)
        user['enabled'] = True
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = self._get_user_ref(user['id'])
        self.assertTrue(user_ref.enabled)

    def test_ignore_user_inactivity(self):
        self.user_dict['options'] = {'ignore_user_inactivity': True}
        user = PROVIDERS.identity_api.create_user(self.user_dict)
        last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(self.max_inactive_days + 1)).date()
        self._update_user_last_active_at(user['id'], last_active_at)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertTrue(user['enabled'])

    def test_ignore_user_inactivity_with_user_disabled(self):
        user = PROVIDERS.identity_api.create_user(self.user_dict)
        last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(self.max_inactive_days + 1)).date()
        self._update_user_last_active_at(user['id'], last_active_at)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertFalse(user['enabled'])
        user['options'] = {'ignore_user_inactivity': True}
        user = PROVIDERS.identity_api.update_user(user['id'], user)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertFalse(user['enabled'])
        user['enabled'] = True
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertTrue(user['enabled'])

    def _get_user_dict(self, password):
        user = {'name': uuid.uuid4().hex, 'domain_id': CONF.identity.default_domain_id, 'enabled': True, 'password': password}
        return user

    def _get_user_ref(self, user_id):
        with sql.session_for_read() as session:
            return session.get(model.User, user_id)

    def _create_user(self, user_dict, last_active_at):
        user_dict['id'] = uuid.uuid4().hex
        with sql.session_for_write() as session:
            user_ref = model.User.from_dict(user_dict)
            user_ref.last_active_at = last_active_at
            session.add(user_ref)
            return base.filter_user(user_ref.to_dict())

    def _update_user_last_active_at(self, user_id, last_active_at):
        with sql.session_for_write() as session:
            user_ref = session.get(model.User, user_id)
            user_ref.last_active_at = last_active_at
            return user_ref