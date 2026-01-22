import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
class TestTrustOperations(test_v3.RestfulTestCase):
    """Test module for create, read, update and delete operations on trusts.

    This module is specific to tests for trust CRUD operations. All other tests
    related to trusts that are authentication or authorization specific should
    live in the keystone/tests/unit/test_v3_auth.py module.

    """

    def setUp(self):
        super(TestTrustOperations, self).setUp()
        self.trustee_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        self.trustee_user_id = self.trustee_user['id']

    def test_create_trust_bad_request(self):
        self.post('/OS-TRUST/trusts', body={'trust': {}}, expected_status=http.client.FORBIDDEN)

    def test_create_trust_with_invalid_expiration_fails(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
        ref['expires_at'] = 'bad'
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)
        ref['expires_at'] = ''
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)
        ref['expires_at'] = 'Z'
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_trusts_do_not_implement_updates(self):
        with self.test_client() as c:
            token = self.get_scoped_token()
            ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
            r = c.post('/v3/OS-TRUST/trusts', json={'trust': ref}, headers={'X-Auth-Token': token})
            trust_id = r.json['trust']['id']
            c.patch('/v3/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust_id}, json={'trust': ref}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
            c.put('/v3/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust_id}, json={'trust': ref}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)

    def test_trust_crud(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        self.assertValidTrustResponse(r, ref)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s/roles' % {'trust_id': trust['id']})
        roles = self.assertValidRoleListResponse(r, self.role)
        self.assertIn(self.role['id'], [x['id'] for x in roles])
        self.head('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']}, expected_status=http.client.OK)
        r = self.get('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']})
        self.assertValidRoleResponse(r, self.role)
        r = self.get('/OS-TRUST/trusts')
        self.assertValidTrustListResponse(r, trust)
        self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']}, expected_status=http.client.NOT_FOUND)

    def test_list_trusts(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        for i in range(3):
            ref['expires_at'] = datetime.datetime.utcnow().replace(year=2032).strftime(unit.TIME_FORMAT)
            r = self.post('/OS-TRUST/trusts', body={'trust': ref})
            self.assertValidTrustResponse(r, ref)
        list_url = '/OS-TRUST/trusts'
        r = self.get(list_url)
        self.head(list_url, expected_status=http.client.OK)
        trusts = r.result['trusts']
        self.assertEqual(3, len(trusts))
        self.assertValidTrustListResponse(r)
        list_for_trustor_url = '/OS-TRUST/trusts?trustor_user_id=%s' % self.user_id
        r = self.get(list_for_trustor_url)
        self.head(list_for_trustor_url, expected_status=http.client.OK)
        trusts = r.result['trusts']
        self.assertEqual(3, len(trusts))
        self.assertValidTrustListResponse(r)
        list_as_trustor_url = '/OS-TRUST/trusts?trustee_user_id=%s' % self.user_id
        r = self.get(list_as_trustor_url)
        self.head(list_as_trustor_url, expected_status=http.client.OK)
        trusts = r.result['trusts']
        self.assertEqual(0, len(trusts))

    def test_create_trust_with_expiration_in_the_past_fails(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, expires='2010-06-04T08:44:31.999999Z', role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_delete_trust(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r, ref)
        self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
        self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']}, expected_status=http.client.NOT_FOUND)

    def test_create_trust_without_trustee_returns_bad_request(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
        del ref['trustee_user_id']
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_trust_without_impersonation_returns_bad_request(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
        del ref['impersonation']
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_trust_with_bad_remaining_uses_returns_bad_request(self):
        for value in [-1, 0, 'a bad value', 7.2]:
            ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, remaining_uses=value, role_ids=[self.role_id])
            self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_trust_with_non_existant_trustee_returns_not_found(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=uuid.uuid4().hex, project_id=self.project_id, role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.NOT_FOUND)

    def test_create_trust_with_trustee_as_trustor_returns_forbidden(self):
        ref = unit.new_trust_ref(trustor_user_id=self.trustee_user_id, trustee_user_id=self.user_id, project_id=self.project_id, role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.FORBIDDEN)

    def test_create_trust_with_non_existant_project_returns_not_found(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=uuid.uuid4().hex, role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.NOT_FOUND)

    def test_create_trust_with_non_existant_role_id_returns_not_found(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[uuid.uuid4().hex])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.NOT_FOUND)

    def test_create_trust_with_extra_attributes_fails(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
        ref['roles'].append({'fake_key': 'fake_value'})
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_create_trust_with_non_existant_role_name_returns_not_found(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_names=[uuid.uuid4().hex])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.NOT_FOUND)

    def test_create_trust_with_role_name_ambiguous_returns_bad_request(self):
        role_ref = unit.new_role_ref(name=self.role['name'], domain_id=uuid.uuid4().hex)
        self.post('/roles', body={'role': role_ref})
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_names=[self.role['name']])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)

    def test_exercise_trust_scoped_token_without_impersonation(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        resp = self.v3_create_token(auth_data)
        resp_body = resp.json_body['token']
        self.assertValidProjectScopedTokenResponse(resp, self.trustee_user)
        self.assertEqual(self.trustee_user['id'], resp_body['user']['id'])
        self.assertEqual(self.trustee_user['name'], resp_body['user']['name'])
        self.assertEqual(self.domain['id'], resp_body['user']['domain']['id'])
        self.assertEqual(self.domain['name'], resp_body['user']['domain']['name'])
        self.assertEqual(self.project['id'], resp_body['project']['id'])
        self.assertEqual(self.project['name'], resp_body['project']['name'])

    def test_exercise_trust_scoped_token_with_impersonation(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
        resp = self.v3_create_token(auth_data)
        resp_body = resp.json_body['token']
        self.assertValidProjectScopedTokenResponse(resp, self.user)
        self.assertEqual(self.user['id'], resp_body['user']['id'])
        self.assertEqual(self.user['name'], resp_body['user']['name'])
        self.assertEqual(self.domain['id'], resp_body['user']['domain']['id'])
        self.assertEqual(self.domain['name'], resp_body['user']['domain']['name'])
        self.assertEqual(self.project['id'], resp_body['project']['id'])
        self.assertEqual(self.project['name'], resp_body['project']['name'])

    def test_forbidden_trust_impersonation_in_redelegation(self):
        """Test forbiddance of impersonation in trust redelegation.

        Check that trustee not allowed to create a trust (with impersonation
        set to true) from a redelegated trust (with impersonation set to false)
        """
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, role_ids=[self.role_id], allow_redelegation=True)
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        auth_data = self.build_authentication_request(user_id=self.trustee_user_id, password=self.trustee_user['password'], trust_id=trust['id'])
        resp = self.v3_create_token(auth_data)
        third_party_trustee = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        third_party_trustee_id = third_party_trustee['id']
        ref = unit.new_trust_ref(trustor_user_id=self.trustee_user_id, trustee_user_id=third_party_trustee_id, project_id=self.project_id, impersonation=True, role_ids=[self.role_id])
        ref['redelegated_trust_id'] = trust['id']
        self.admin_request(path='/v3/OS-TRUST/trusts', body={'trust': ref}, token=resp.headers.get('X-Subject-Token'), method='POST', expected_status=http.client.FORBIDDEN)

    def test_trust_deleted_when_user_deleted(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, role_ids=[self.role_id], allow_redelegation=True)
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        r = self.get('/OS-TRUST/trusts')
        self.assertEqual(1, len(r.result['trusts']))
        self.delete('/users/%(user_id)s' % {'user_id': trust['trustee_user_id']})
        self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']}, expected_status=http.client.NOT_FOUND)
        trustee_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        trustee_user_id = trustee_user['id']
        ref['trustee_user_id'] = trustee_user_id
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        r = self.get('/OS-TRUST/trusts')
        self.assertEqual(1, len(r.result['trusts']))
        self.delete('/users/%(user_id)s' % {'user_id': trust['trustor_user_id']})
        self.assertRaises(exception.TrustNotFound, PROVIDERS.trust_api.get_trust, trust['id'])

    def test_trust_deleted_when_project_deleted(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, role_ids=[self.role_id], allow_redelegation=True)
        resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(resp)
        r = self.get('/OS-TRUST/trusts')
        self.assertEqual(1, len(r.result['trusts']))
        self.delete('/projects/%(project_id)s' % {'project_id': trust['project_id']})
        self.assertRaises(exception.TrustNotFound, PROVIDERS.trust_api.get_trust, trust['id'])