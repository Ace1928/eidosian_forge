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
class PasswordHistoryValidationTests(test_backend_sql.SqlTests):

    def setUp(self):
        super(PasswordHistoryValidationTests, self).setUp()
        self.max_cnt = 3
        self.config_fixture.config(group='security_compliance', unique_last_password_count=self.max_cnt)

    def test_validate_password_history_with_invalid_password(self):
        password = uuid.uuid4().hex
        user = self._create_user(password)
        with self.make_request():
            self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user['id'], original_password=password, new_password=password)
            new_password = uuid.uuid4().hex
            self.assertValidChangePassword(user['id'], password, new_password)
            self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user['id'], original_password=new_password, new_password=password)

    def test_validate_password_history_with_valid_password(self):
        passwords = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex]
        user = self._create_user(passwords[0])
        self.assertValidChangePassword(user['id'], passwords[0], passwords[1])
        self.assertValidChangePassword(user['id'], passwords[1], passwords[2])
        self.assertValidChangePassword(user['id'], passwords[2], passwords[3])
        self.assertValidChangePassword(user['id'], passwords[3], passwords[0])

    def test_validate_password_history_with_valid_password_only_once(self):
        self.config_fixture.config(group='security_compliance', unique_last_password_count=1)
        passwords = [uuid.uuid4().hex, uuid.uuid4().hex]
        user = self._create_user(passwords[0])
        self.assertValidChangePassword(user['id'], passwords[0], passwords[1])
        self.assertValidChangePassword(user['id'], passwords[1], passwords[0])

    def test_validate_password_history_but_start_with_password_none(self):
        passwords = [uuid.uuid4().hex, uuid.uuid4().hex]
        user = self._create_user(None)
        user_ref = self._get_user_ref(user['id'])
        self.assertIsNone(user_ref.password)
        user['password'] = passwords[0]
        PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertValidChangePassword(user['id'], passwords[0], passwords[1])
        with self.make_request():
            self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user['id'], original_password=passwords[1], new_password=passwords[0])

    def test_disable_password_history_and_repeat_same_password(self):
        self.config_fixture.config(group='security_compliance', unique_last_password_count=0)
        password = uuid.uuid4().hex
        user = self._create_user(password)
        self.assertValidChangePassword(user['id'], password, password)
        self.assertValidChangePassword(user['id'], password, password)

    def test_admin_password_reset_is_not_validated_by_password_history(self):
        passwords = [uuid.uuid4().hex, uuid.uuid4().hex]
        user = self._create_user(passwords[0])
        user['password'] = passwords[1]
        with self.make_request():
            PROVIDERS.identity_api.update_user(user['id'], user)
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=passwords[1])
            user['password'] = passwords[1]
            PROVIDERS.identity_api.update_user(user['id'], user)
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=passwords[1])
            user['password'] = passwords[0]
            PROVIDERS.identity_api.update_user(user['id'], user)
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=passwords[0])

    def test_truncate_passwords(self):
        user = self._create_user(uuid.uuid4().hex)
        self._add_passwords_to_history(user, n=4)
        user_ref = self._get_user_ref(user['id'])
        self.assertEqual(len(user_ref.local_user.passwords), self.max_cnt + 1)

    def test_truncate_passwords_when_max_is_default(self):
        self.max_cnt = 1
        expected_length = self.max_cnt + 1
        self.config_fixture.config(group='security_compliance', unique_last_password_count=self.max_cnt)
        user = self._create_user(uuid.uuid4().hex)
        self._add_passwords_to_history(user, n=4)
        user_ref = self._get_user_ref(user['id'])
        self.assertEqual(len(user_ref.local_user.passwords), expected_length)
        self.max_cnt = 4
        self.config_fixture.config(group='security_compliance', unique_last_password_count=self.max_cnt)
        self._add_passwords_to_history(user, n=self.max_cnt)
        user_ref = self._get_user_ref(user['id'])
        self.assertEqual(len(user_ref.local_user.passwords), self.max_cnt + 1)
        self.max_cnt = 1
        self.config_fixture.config(group='security_compliance', unique_last_password_count=self.max_cnt)
        self._add_passwords_to_history(user, n=1)
        user_ref = self._get_user_ref(user['id'])
        self.assertEqual(len(user_ref.local_user.passwords), expected_length)

    def test_truncate_passwords_when_max_is_default_and_no_password(self):
        expected_length = 1
        self.max_cnt = 1
        self.config_fixture.config(group='security_compliance', unique_last_password_count=self.max_cnt)
        user = {'name': uuid.uuid4().hex, 'domain_id': 'default', 'enabled': True}
        user = PROVIDERS.identity_api.create_user(user)
        self._add_passwords_to_history(user, n=1)
        user_ref = self._get_user_ref(user['id'])
        self.assertEqual(len(user_ref.local_user.passwords), expected_length)

    def _create_user(self, password):
        user = {'name': uuid.uuid4().hex, 'domain_id': 'default', 'enabled': True, 'password': password}
        return PROVIDERS.identity_api.create_user(user)

    def assertValidChangePassword(self, user_id, password, new_password):
        with self.make_request():
            PROVIDERS.identity_api.change_password(user_id=user_id, original_password=password, new_password=new_password)
            PROVIDERS.identity_api.authenticate(user_id=user_id, password=new_password)

    def _add_passwords_to_history(self, user, n):
        for _ in range(n):
            user['password'] = uuid.uuid4().hex
            PROVIDERS.identity_api.update_user(user['id'], user)

    def _get_user_ref(self, user_id):
        with sql.session_for_read() as session:
            return PROVIDERS.identity_api._get_user(session, user_id)