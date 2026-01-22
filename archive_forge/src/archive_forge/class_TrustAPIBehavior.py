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
class TrustAPIBehavior(test_v3.RestfulTestCase):
    """Redelegation valid and secure.

    Redelegation is a hierarchical structure of trusts between initial trustor
    and a group of users allowed to impersonate trustor and act in his name.
    Hierarchy is created in a process of trusting already trusted permissions
    and organized as an adjacency list using 'redelegated_trust_id' field.
    Redelegation is valid if each subsequent trust in a chain passes 'not more'
    permissions than being redelegated.

    Trust constraints are:
     * roles - set of roles trusted by trustor
     * expiration_time
     * allow_redelegation - a flag
     * redelegation_count - decreasing value restricting length of trust chain
     * remaining_uses - DISALLOWED when allow_redelegation == True

    Trust becomes invalid in case:
     * trust roles were revoked from trustor
     * one of the users in the delegation chain was disabled or deleted
     * expiration time passed
     * one of the parent trusts has become invalid
     * one of the parent trusts was deleted

    """

    def config_overrides(self):
        super(TrustAPIBehavior, self).config_overrides()
        self.config_fixture.config(group='trust', allow_redelegation=True, max_redelegation_count=10)

    def setUp(self):
        super(TrustAPIBehavior, self).setUp()
        self.trustee_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        self.redelegated_trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id], allow_redelegation=True)
        self.chained_trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, role_ids=[self.role_id], allow_redelegation=True)

    def _get_trust_token(self, trust):
        trust_id = trust['id']
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust_id)
        trust_token = self.get_requested_token(auth_data)
        return trust_token

    def test_depleted_redelegation_count_error(self):
        self.redelegated_trust_ref['redelegation_count'] = 0
        r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_modified_redelegation_count_error(self):
        r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        correct = trust['redelegation_count'] - 1
        incorrect = correct - 1
        self.chained_trust_ref['redelegation_count'] = incorrect
        self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_max_redelegation_count_constraint(self):
        incorrect = CONF.trust.max_redelegation_count + 1
        self.redelegated_trust_ref['redelegation_count'] = incorrect
        self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref}, expected_status=http.client.FORBIDDEN)

    def test_redelegation_expiry(self):
        r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        too_long_live_chained_trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=10), role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': too_long_live_chained_trust_ref}, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_redelegation_remaining_uses(self):
        r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        self.chained_trust_ref['remaining_uses'] = 5
        self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token, expected_status=http.client.BAD_REQUEST)

    def test_roles_subset(self):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        PROVIDERS.assignment_api.create_grant(role_id=role['id'], user_id=self.user_id, project_id=self.project_id)
        ref = self.redelegated_trust_ref
        ref['expires_at'] = datetime.datetime.utcnow().replace(year=2032).strftime(unit.TIME_FORMAT)
        ref['roles'].append({'id': role['id']})
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        role_id_set = set((r['id'] for r in ref['roles']))
        trust_role_id_set = set((r['id'] for r in trust['roles']))
        self.assertEqual(role_id_set, trust_role_id_set)
        trust_token = self._get_trust_token(trust)
        self.chained_trust_ref['expires_at'] = datetime.datetime.utcnow().replace(year=2028).strftime(unit.TIME_FORMAT)
        r = self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token)
        trust2 = self.assertValidTrustResponse(r)
        role_id_set1 = set((r['id'] for r in trust['roles']))
        role_id_set2 = set((r['id'] for r in trust2['roles']))
        self.assertThat(role_id_set1, matchers.GreaterThan(role_id_set2))

    def test_trust_with_implied_roles(self):
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        role2 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role2['id'], role2)
        PROVIDERS.role_api.create_implied_role(role1['id'], role2['id'])
        PROVIDERS.assignment_api.create_grant(role_id=role1['id'], user_id=self.user_id, project_id=self.project_id)
        ref = self.redelegated_trust_ref
        ref['roles'] = [{'id': role1['id']}, {'id': role2['id']}]
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        role_ids = [r['id'] for r in ref['roles']]
        trust_role_ids = [r['id'] for r in trust['roles']]
        self.assertEqual(role_ids, trust_role_ids)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        resp = self.post('/auth/tokens', body=auth_data)
        trust_token_role_ids = [r['id'] for r in resp.json['token']['roles']]
        self.assertEqual(sorted(role_ids), sorted(trust_token_role_ids))

    def test_redelegate_with_role_by_name(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_names=[self.role['name']], allow_redelegation=True)
        ref['expires_at'] = datetime.datetime.utcnow().replace(year=2032).strftime(unit.TIME_FORMAT)
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, role_names=[self.role['name']], allow_redelegation=True)
        ref['expires_at'] = datetime.datetime.utcnow().replace(year=2028).strftime(unit.TIME_FORMAT)
        r = self.post('/OS-TRUST/trusts', body={'trust': ref}, token=trust_token)
        trust = self.assertValidTrustResponse(r)
        self._get_trust_token(trust)

    def test_redelegate_new_role_fails(self):
        r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        PROVIDERS.assignment_api.create_grant(role_id=role['id'], user_id=self.user_id, project_id=self.project_id)
        self.chained_trust_ref['roles'] = [{'id': role['id']}]
        with mock.patch.object(policy, 'enforce', return_value=True):
            self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_redelegation_terminator(self):
        self.redelegated_trust_ref['expires_at'] = datetime.datetime.utcnow().replace(year=2032).strftime(unit.TIME_FORMAT)
        r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(r)
        trust_token = self._get_trust_token(trust)
        self.chained_trust_ref['expires_at'] = datetime.datetime.utcnow().replace(year=2028).strftime(unit.TIME_FORMAT)
        ref = dict(self.chained_trust_ref, redelegation_count=1, allow_redelegation=False)
        r = self.post('/OS-TRUST/trusts', body={'trust': ref}, token=trust_token)
        trust = self.assertValidTrustResponse(r)
        self.assertNotIn('allow_redelegation', trust)
        self.assertEqual(0, trust['redelegation_count'])
        trust_token = self._get_trust_token(trust)
        self.post('/OS-TRUST/trusts', body={'trust': ref}, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_redelegation_without_impersonation(self):
        self.redelegated_trust_ref['impersonation'] = False
        resp = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref}, expected_status=http.client.CREATED)
        trust = self.assertValidTrustResponse(resp)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        trust_token = self.get_requested_token(auth_data)
        trustee_user_2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        trust_ref_2 = unit.new_trust_ref(trustor_user_id=self.trustee_user['id'], trustee_user_id=trustee_user_2['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id], allow_redelegation=False)
        resp = self.post('/OS-TRUST/trusts', body={'trust': trust_ref_2}, token=trust_token, expected_status=http.client.NOT_FOUND)

    def test_create_unscoped_trust(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        self.assertValidTrustResponse(r, ref)

    def test_create_trust_no_roles(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id)
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.FORBIDDEN)

    def _initialize_test_consume_trust(self, count):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, remaining_uses=count, role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'])
        r = self.v3_create_token(auth_data)
        token = r.headers.get('X-Subject-Token')
        auth_data = self.build_authentication_request(token=token, trust_id=trust['id'])
        r = self.v3_create_token(auth_data)
        return trust

    def test_authenticate_without_trust_dict_returns_bad_request(self):
        token = self.v3_create_token(self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'])).headers.get('X-Subject-Token')
        auth_data = {'auth': {'identity': {'methods': ['token'], 'token': {'id': token}}, 'scope': {'OS-TRUST:trust': ''}}}
        self.admin_request(method='POST', path='/v3/auth/tokens', body=auth_data, expected_status=http.client.BAD_REQUEST)

    def test_consume_trust_once(self):
        trust = self._initialize_test_consume_trust(2)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        trust = r.result.get('trust')
        self.assertIsNotNone(trust)
        self.assertEqual(1, trust['remaining_uses'])
        self.assertEqual(self.role['name'], trust['roles'][0]['name'])
        self.assertEqual(self.role['id'], trust['roles'][0]['id'])

    def test_create_one_time_use_trust(self):
        trust = self._initialize_test_consume_trust(1)
        self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']}, expected_status=http.client.NOT_FOUND)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_unlimited_use_trust(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, remaining_uses=None, role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'])
        r = self.v3_create_token(auth_data)
        token = r.headers.get('X-Subject-Token')
        auth_data = self.build_authentication_request(token=token, trust_id=trust['id'])
        r = self.v3_create_token(auth_data)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        trust = r.result.get('trust')
        self.assertIsNone(trust['remaining_uses'])

    def test_impersonation_token_cannot_create_new_trust(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        trust_token = self.get_requested_token(auth_data)
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_trust_deleted_grant(self):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        grant_url = '/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'user_id': self.user_id, 'role_id': role['id']}
        self.put(grant_url)
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[role['id']])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        self.delete(grant_url)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        r = self.v3_create_token(auth_data, expected_status=http.client.FORBIDDEN)

    def test_trust_chained(self):
        """Test that a trust token can't be used to execute another trust.

        To do this, we create an A->B->C hierarchy of trusts, then attempt to
        execute the trusts in series (C->B->A).

        """
        sub_trustee_user = unit.create_user(PROVIDERS.identity_api, domain_id=test_v3.DEFAULT_DOMAIN_ID)
        sub_trustee_user_id = sub_trustee_user['id']
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        self.put('/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'user_id': self.trustee_user['id'], 'role_id': role['id']})
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust1 = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], project_id=self.project_id)
        token = self.get_requested_token(auth_data)
        ref = unit.new_trust_ref(trustor_user_id=self.trustee_user['id'], trustee_user_id=sub_trustee_user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[role['id']])
        r = self.post('/OS-TRUST/trusts', token=token, body={'trust': ref})
        trust2 = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=sub_trustee_user['id'], password=sub_trustee_user['password'], trust_id=trust2['id'])
        trust_token = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(token=trust_token, trust_id=trust1['id'])
        r = self.v3_create_token(auth_data, expected_status=http.client.FORBIDDEN)

    def assertTrustTokensRevoked(self, trust_id):
        revocation_response = self.get('/OS-REVOKE/events')
        revocation_events = revocation_response.json_body['events']
        found = False
        for event in revocation_events:
            if event.get('OS-TRUST:trust_id') == trust_id:
                found = True
        self.assertTrue(found, 'event with trust_id %s not found in list' % trust_id)

    def test_delete_trust_revokes_tokens(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        trust_id = trust['id']
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust_id)
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r, self.trustee_user)
        trust_token = r.headers['X-Subject-Token']
        self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust_id})
        headers = {'X-Subject-Token': trust_token}
        self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
        self.assertTrustTokensRevoked(trust_id)

    def disable_user(self, user):
        user['enabled'] = False
        PROVIDERS.identity_api.update_user(user['id'], user)

    def test_trust_get_token_fails_if_trustor_disabled(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        self.v3_create_token(auth_data)
        self.disable_user(self.user)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        self.v3_create_token(auth_data, expected_status=http.client.FORBIDDEN)

    def test_trust_get_token_fails_if_trustee_disabled(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        self.v3_create_token(auth_data)
        self.disable_user(self.trustee_user)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_delete_trust(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_change_password_invalidates_trust_tokens(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r, self.user)
        trust_token = r.headers.get('X-Subject-Token')
        self.get('/OS-TRUST/trusts?trustor_user_id=%s' % self.user_id, token=trust_token)
        self.assertValidUserResponse(self.patch('/users/%s' % self.trustee_user['id'], body={'user': {'password': uuid.uuid4().hex}}))
        self.get('/OS-TRUST/trusts?trustor_user_id=%s' % self.user_id, expected_status=http.client.UNAUTHORIZED, token=trust_token)

    def test_trustee_can_do_role_ops(self):
        resp = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
        trust = self.assertValidTrustResponse(resp)
        trust_token = self._get_trust_token(trust)
        resp = self.get('/OS-TRUST/trusts/%(trust_id)s/roles' % {'trust_id': trust['id']}, token=trust_token)
        self.assertValidRoleListResponse(resp, self.role)
        self.head('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']}, token=trust_token, expected_status=http.client.OK)
        resp = self.get('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']}, token=trust_token)
        self.assertValidRoleResponse(resp, self.role)

    def test_do_not_consume_remaining_uses_when_get_token_fails(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id], remaining_uses=3)
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        new_trust = r.result.get('trust')
        trust_id = new_trust.get('id')
        auth_data = self.build_authentication_request(user_id=self.default_domain_user['id'], password=self.default_domain_user['password'], trust_id=trust_id)
        self.v3_create_token(auth_data, expected_status=http.client.FORBIDDEN)
        r = self.get('/OS-TRUST/trusts/%s' % trust_id)
        self.assertEqual(3, r.result.get('trust').get('remaining_uses'))