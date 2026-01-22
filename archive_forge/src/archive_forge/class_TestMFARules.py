import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestMFARules(test_v3.RestfulTestCase):

    def config_overrides(self):
        super(TestMFARules, self).config_overrides()
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_fernet.MAX_ACTIVE_KEYS))

    def assertValidErrorResponse(self, r):
        resp = r.result
        if r.headers.get(authorization.AUTH_RECEIPT_HEADER):
            self.assertIsNotNone(resp.get('receipt'))
            self.assertIsNotNone(resp.get('receipt').get('methods'))
        else:
            self.assertIsNotNone(resp.get('error'))
            self.assertIsNotNone(resp['error'].get('code'))
            self.assertIsNotNone(resp['error'].get('title'))
            self.assertIsNotNone(resp['error'].get('message'))
            self.assertEqual(int(resp['error']['code']), r.status_code)

    def _create_totp_cred(self):
        totp_cred = unit.new_totp_credential(self.user_id, self.project_id)
        PROVIDERS.credential_api.create_credential(uuid.uuid4().hex, totp_cred)

        def cleanup(testcase):
            totp_creds = testcase.credential_api.list_credentials_for_user(testcase.user['id'], type='totp')
            for cred in totp_creds:
                testcase.credential_api.delete_credential(cred['id'])
        self.addCleanup(cleanup, testcase=self)
        return totp_cred

    def auth_plugin_config_override(self, methods=None, **method_classes):
        methods = ['totp', 'token', 'password']
        super(TestMFARules, self).auth_plugin_config_override(methods)

    def _update_user_with_MFA_rules(self, rule_list, rules_enabled=True):
        user = self.user.copy()
        user.pop('password')
        user['options'][ro.MFA_RULES_OPT.option_name] = rule_list
        user['options'][ro.MFA_ENABLED_OPT.option_name] = rules_enabled
        PROVIDERS.identity_api.update_user(user['id'], user)

    def test_MFA_single_method_rules_requirements_met_succeeds(self):
        rule_list = [['password'], ['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            self.v3_create_token(self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id))

    def test_MFA_multi_method_rules_requirements_met_succeeds(self):
        rule_list = [['password', 'totp']]
        totp_cred = self._create_totp_cred()
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            auth_req = self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, passcode=totp._generate_totp_passcodes(totp_cred['blob'])[0])
            self.v3_create_token(auth_req)

    def test_MFA_single_method_rules_requirements_not_met_fails(self):
        rule_list = [['totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            self.v3_create_token(self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id), expected_status=http.client.UNAUTHORIZED)

    def test_MFA_multi_method_rules_requirements_not_met_fails(self):
        rule_list = [['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            self.v3_create_token(self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id), expected_status=http.client.UNAUTHORIZED)

    def test_MFA_rules_bogus_non_existing_auth_method_succeeds(self):
        rule_list = [['password'], ['BoGusAuThMeTh0dHandl3r']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            self.v3_create_token(self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id))

    def test_MFA_rules_disabled_MFA_succeeeds(self):
        rule_list = [['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list, rules_enabled=False)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            self.v3_create_token(self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id))

    def test_MFA_rules_all_bogus_rules_results_in_default_behavior(self):
        rule_list = [[uuid.uuid4().hex, uuid.uuid4().hex], ['BoGus'], ['NonExistantMethod']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            self.v3_create_token(self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id))

    def test_MFA_rules_rescope_works_without_token_method_in_rules(self):
        rule_list = [['password', 'totp']]
        totp_cred = self._create_totp_cred()
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            auth_data = self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, passcode=totp._generate_totp_passcodes(totp_cred['blob'])[0])
            r = self.v3_create_token(auth_data)
            auth_data = self.build_authentication_request(token=r.headers.get('X-Subject-Token'), project_id=self.project_id)
            self.v3_create_token(auth_data)

    def test_MFA_requirements_makes_correct_receipt_for_password(self):
        rule_list = [['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', body=self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id), expected_status=http.client.UNAUTHORIZED)
        self.assertIsNotNone(response.headers.get(authorization.AUTH_RECEIPT_HEADER))
        resp_data = response.result
        self.assertEqual({'password'}, set(resp_data.get('receipt').get('methods')))
        self.assertEqual(set((frozenset(r) for r in rule_list)), set((frozenset(r) for r in resp_data.get('required_auth_methods'))))

    def test_MFA_requirements_makes_correct_receipt_for_totp(self):
        totp_cred = self._create_totp_cred()
        rule_list = [['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', body=self.build_authentication_request(user_id=self.user_id, user_domain_id=self.domain_id, project_id=self.project_id, passcode=totp._generate_totp_passcodes(totp_cred['blob'])[0]), expected_status=http.client.UNAUTHORIZED)
        self.assertIsNotNone(response.headers.get(authorization.AUTH_RECEIPT_HEADER))
        resp_data = response.result
        self.assertEqual({'totp'}, set(resp_data.get('receipt').get('methods')))
        self.assertEqual(set((frozenset(r) for r in rule_list)), set((frozenset(r) for r in resp_data.get('required_auth_methods'))))

    def test_MFA_requirements_makes_correct_receipt_for_pass_and_totp(self):
        totp_cred = self._create_totp_cred()
        rule_list = [['password', 'totp', 'token']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', body=self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id, passcode=totp._generate_totp_passcodes(totp_cred['blob'])[0]), expected_status=http.client.UNAUTHORIZED)
        self.assertIsNotNone(response.headers.get(authorization.AUTH_RECEIPT_HEADER))
        resp_data = response.result
        self.assertEqual({'password', 'totp'}, set(resp_data.get('receipt').get('methods')))
        self.assertEqual(set((frozenset(r) for r in rule_list)), set((frozenset(r) for r in resp_data.get('required_auth_methods'))))

    def test_MFA_requirements_returns_correct_required_auth_methods(self):
        rule_list = [['password', 'totp', 'token'], ['password', 'totp'], ['token', 'totp'], ['BoGusAuThMeTh0dHandl3r']]
        expect_rule_list = rule_list = [['password', 'totp', 'token'], ['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', body=self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id), expected_status=http.client.UNAUTHORIZED)
        self.assertIsNotNone(response.headers.get(authorization.AUTH_RECEIPT_HEADER))
        resp_data = response.result
        self.assertEqual({'password'}, set(resp_data.get('receipt').get('methods')))
        self.assertEqual(set((frozenset(r) for r in expect_rule_list)), set((frozenset(r) for r in resp_data.get('required_auth_methods'))))

    def test_MFA_consuming_receipt_with_totp(self):
        totp_cred = self._create_totp_cred()
        rule_list = [['password', 'totp']]
        self._update_user_with_MFA_rules(rule_list=rule_list)
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', body=self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, project_id=self.project_id), expected_status=http.client.UNAUTHORIZED)
        self.assertIsNotNone(response.headers.get(authorization.AUTH_RECEIPT_HEADER))
        receipt = response.headers.get(authorization.AUTH_RECEIPT_HEADER)
        resp_data = response.result
        self.assertEqual({'password'}, set(resp_data.get('receipt').get('methods')))
        self.assertEqual(set((frozenset(r) for r in rule_list)), set((frozenset(r) for r in resp_data.get('required_auth_methods'))))
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', headers={authorization.AUTH_RECEIPT_HEADER: receipt}, body=self.build_authentication_request(user_id=self.user_id, user_domain_id=self.domain_id, project_id=self.project_id, passcode=totp._generate_totp_passcodes(totp_cred['blob'])[0]))

    def test_MFA_consuming_receipt_not_found(self):
        time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
        with freezegun.freeze_time(time):
            response = self.admin_request(method='POST', path='/v3/auth/tokens', headers={authorization.AUTH_RECEIPT_HEADER: 'bogus-receipt'}, body=self.build_authentication_request(user_id=self.user_id, user_domain_id=self.domain_id, project_id=self.project_id), expected_status=http.client.UNAUTHORIZED)
        self.assertEqual(401, response.result['error']['code'])