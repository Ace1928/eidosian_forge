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
class LockingOutUserTests(test_backend_sql.SqlTests):

    def setUp(self):
        super(LockingOutUserTests, self).setUp()
        self.config_fixture.config(group='security_compliance', lockout_failure_attempts=6)
        self.config_fixture.config(group='security_compliance', lockout_duration=5)
        self.password = uuid.uuid4().hex
        user_dict = {'name': uuid.uuid4().hex, 'domain_id': CONF.identity.default_domain_id, 'enabled': True, 'password': self.password}
        self.user = PROVIDERS.identity_api.create_user(user_dict)

    def test_locking_out_user_after_max_failed_attempts(self):
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
            PROVIDERS.identity_api.authenticate(user_id=self.user['id'], password=self.password)
            self._fail_auth_repeatedly(self.user['id'])
            self.assertRaises(exception.Unauthorized, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)

    def test_lock_out_for_ignored_user(self):
        self.user['options'][iro.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name] = True
        PROVIDERS.identity_api.update_user(self.user['id'], self.user)
        self._fail_auth_repeatedly(self.user['id'])
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
            PROVIDERS.identity_api.authenticate(user_id=self.user['id'], password=self.password)

    def test_set_enabled_unlocks_user(self):
        with self.make_request():
            self._fail_auth_repeatedly(self.user['id'])
            self.assertRaises(exception.Unauthorized, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
            self.user['enabled'] = True
            PROVIDERS.identity_api.update_user(self.user['id'], self.user)
            user_ret = PROVIDERS.identity_api.authenticate(user_id=self.user['id'], password=self.password)
            self.assertTrue(user_ret['enabled'])

    def test_lockout_duration(self):
        with freezegun.freeze_time(datetime.datetime.utcnow()) as frozen_time:
            with self.make_request():
                self._fail_auth_repeatedly(self.user['id'])
                self.assertRaises(exception.Unauthorized, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
                frozen_time.tick(delta=datetime.timedelta(seconds=CONF.security_compliance.lockout_duration + 1))
                PROVIDERS.identity_api.authenticate(user_id=self.user['id'], password=self.password)
                self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)

    def test_lockout_duration_failed_auth_cnt_resets(self):
        with freezegun.freeze_time(datetime.datetime.utcnow()) as frozen_time:
            with self.make_request():
                self._fail_auth_repeatedly(self.user['id'])
                self.assertRaises(exception.Unauthorized, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
                frozen_time.tick(delta=datetime.timedelta(seconds=CONF.security_compliance.lockout_duration + 1))
                self._fail_auth_repeatedly(self.user['id'])
                self.assertRaises(exception.Unauthorized, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)

    def _fail_auth_repeatedly(self, user_id):
        wrong_password = uuid.uuid4().hex
        for _ in range(CONF.security_compliance.lockout_failure_attempts):
            with self.make_request():
                self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user_id, password=wrong_password)