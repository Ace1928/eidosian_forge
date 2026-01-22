import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
class ImpliedRolesTests(test_v3.RestfulTestCase, test_v3.AssignmentTestMixin, unit.TestCase):

    def _create_role(self):
        """Call ``POST /roles``."""
        ref = unit.new_role_ref()
        r = self.post('/roles', body={'role': ref})
        return self.assertValidRoleResponse(r, ref)

    def test_list_implied_roles_none(self):
        self.prior = self._create_role()
        url = '/roles/%s/implies' % self.prior['id']
        response = self.get(url).json['role_inference']
        self.head(url, expected_status=http.client.OK)
        self.assertEqual(self.prior['id'], response['prior_role']['id'])
        self.assertEqual(0, len(response['implies']))

    def _create_implied_role(self, prior, implied):
        self.put('/roles/%s/implies/%s' % (prior['id'], implied['id']), expected_status=http.client.CREATED)

    def _delete_implied_role(self, prior, implied):
        self.delete('/roles/%s/implies/%s' % (prior['id'], implied['id']))

    def _setup_prior_two_implied(self):
        self.prior = self._create_role()
        self.implied1 = self._create_role()
        self._create_implied_role(self.prior, self.implied1)
        self.implied2 = self._create_role()
        self._create_implied_role(self.prior, self.implied2)

    def _assert_expected_implied_role_response(self, expected_prior_id, expected_implied_ids):
        r = self.get('/roles/%s/implies' % expected_prior_id)
        response = r.json
        role_inference = response['role_inference']
        self.assertEqual(expected_prior_id, role_inference['prior_role']['id'])
        prior_link = '/v3/roles/' + expected_prior_id + '/implies'
        self.assertThat(response['links']['self'], matchers.EndsWith(prior_link))
        actual_implied_ids = [implied['id'] for implied in role_inference['implies']]
        self.assertCountEqual(expected_implied_ids, actual_implied_ids)
        self.assertIsNotNone(role_inference['prior_role']['links']['self'])
        for implied in role_inference['implies']:
            self.assertIsNotNone(implied['links']['self'])

    def _assert_expected_role_inference_rule_response(self, expected_prior_id, expected_implied_id):
        url = '/roles/%s/implies/%s' % (expected_prior_id, expected_implied_id)
        response = self.get(url).json
        self.assertThat(response['links']['self'], matchers.EndsWith('/v3%s' % url))
        role_inference = response['role_inference']
        prior_role = role_inference['prior_role']
        self.assertEqual(expected_prior_id, prior_role['id'])
        self.assertIsNotNone(prior_role['name'])
        self.assertThat(prior_role['links']['self'], matchers.EndsWith('/v3/roles/%s' % expected_prior_id))
        implied_role = role_inference['implies']
        self.assertEqual(expected_implied_id, implied_role['id'])
        self.assertIsNotNone(implied_role['name'])
        self.assertThat(implied_role['links']['self'], matchers.EndsWith('/v3/roles/%s' % expected_implied_id))

    def _assert_two_roles_implied(self):
        self._assert_expected_implied_role_response(self.prior['id'], [self.implied1['id'], self.implied2['id']])
        self._assert_expected_role_inference_rule_response(self.prior['id'], self.implied1['id'])
        self._assert_expected_role_inference_rule_response(self.prior['id'], self.implied2['id'])

    def _assert_one_role_implied(self):
        self._assert_expected_implied_role_response(self.prior['id'], [self.implied1['id']])
        self.get('/roles/%s/implies/%s' % (self.prior['id'], self.implied2['id']), expected_status=http.client.NOT_FOUND)

    def _assert_two_rules_defined(self):
        r = self.get('/role_inferences/')
        rules = r.result['role_inferences']
        self.assertEqual(self.prior['id'], rules[0]['prior_role']['id'])
        self.assertEqual(2, len(rules[0]['implies']))
        implied_ids = [implied['id'] for implied in rules[0]['implies']]
        implied_names = [implied['name'] for implied in rules[0]['implies']]
        self.assertIn(self.implied1['id'], implied_ids)
        self.assertIn(self.implied2['id'], implied_ids)
        self.assertIn(self.implied1['name'], implied_names)
        self.assertIn(self.implied2['name'], implied_names)

    def _assert_one_rule_defined(self):
        r = self.get('/role_inferences/')
        rules = r.result['role_inferences']
        self.assertEqual(self.prior['id'], rules[0]['prior_role']['id'])
        self.assertEqual(self.implied1['id'], rules[0]['implies'][0]['id'])
        self.assertEqual(self.implied1['name'], rules[0]['implies'][0]['name'])
        self.assertEqual(1, len(rules[0]['implies']))

    def test_list_all_rules(self):
        self._setup_prior_two_implied()
        self._assert_two_rules_defined()
        self._delete_implied_role(self.prior, self.implied2)
        self._assert_one_rule_defined()

    def test_CRD_implied_roles(self):
        self._setup_prior_two_implied()
        self._assert_two_roles_implied()
        self._delete_implied_role(self.prior, self.implied2)
        self._assert_one_role_implied()

    def _create_three_roles(self):
        self.role_list = []
        for _ in range(3):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            self.role_list.append(role)

    def _create_test_domain_user_project(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        return (domain, user, project)

    def _assign_top_role_to_user_on_project(self, user, project):
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], project['id'], self.role_list[0]['id'])

    def _build_effective_role_assignments_url(self, user):
        return '/role_assignments?effective&user.id=%(user_id)s' % {'user_id': user['id']}

    def _assert_all_roles_in_assignment(self, response, user):
        self.assertValidRoleAssignmentListResponse(response, expected_length=len(self.role_list), resource_url=self._build_effective_role_assignments_url(user))

    def _assert_initial_assignment_in_effective(self, response, user, project):
        entity = self.build_role_assignment_entity(project_id=project['id'], user_id=user['id'], role_id=self.role_list[0]['id'])
        self.assertRoleAssignmentInListResponse(response, entity)

    def _assert_effective_role_for_implied_has_prior_in_links(self, response, user, project, prior_index, implied_index):
        prior_link = '/prior_roles/%(prior)s/implies/%(implied)s' % {'prior': self.role_list[prior_index]['id'], 'implied': self.role_list[implied_index]['id']}
        link = self.build_role_assignment_link(project_id=project['id'], user_id=user['id'], role_id=self.role_list[prior_index]['id'])
        entity = self.build_role_assignment_entity(link=link, project_id=project['id'], user_id=user['id'], role_id=self.role_list[implied_index]['id'], prior_link=prior_link)
        self.assertRoleAssignmentInListResponse(response, entity)

    def test_list_role_assignments_with_implied_roles(self):
        """Call ``GET /role_assignments`` with implied role grant.

        Test Plan:

        - Create a domain with a user and a project
        - Create 3 roles
        - Role 0 implies role 1 and role 1 implies role 2
        - Assign the top role to the project
        - Issue the URL to check effective roles on project - this
          should return all 3 roles.
        - Check the links of the 3 roles indicate the prior role where
          appropriate

        """
        domain, user, project = self._create_test_domain_user_project()
        self._create_three_roles()
        self._create_implied_role(self.role_list[0], self.role_list[1])
        self._create_implied_role(self.role_list[1], self.role_list[2])
        self._assign_top_role_to_user_on_project(user, project)
        response = self.get(self._build_effective_role_assignments_url(user))
        r = response
        self._assert_all_roles_in_assignment(r, user)
        self._assert_initial_assignment_in_effective(response, user, project)
        self._assert_effective_role_for_implied_has_prior_in_links(response, user, project, 0, 1)
        self._assert_effective_role_for_implied_has_prior_in_links(response, user, project, 1, 2)

    def _create_named_role(self, name):
        role = unit.new_role_ref()
        role['name'] = name
        PROVIDERS.role_api.create_role(role['id'], role)
        return role

    def test_root_role_as_implied_role_forbidden(self):
        """Test root role is forbidden to be set as an implied role.

        Create 2 roles that are prohibited from being an implied role.
        Create 1 additional role which should be accepted as an implied
        role. Assure the prohibited role names cannot be set as an implied
        role. Assure the accepted role name which is not a member of the
        prohibited implied role list can be successfully set an implied
        role.
        """
        prohibited_name1 = 'root1'
        prohibited_name2 = 'root2'
        accepted_name1 = 'implied1'
        prohibited_names = [prohibited_name1, prohibited_name2]
        self.config_fixture.config(group='assignment', prohibited_implied_role=prohibited_names)
        prior_role = self._create_role()
        prohibited_role1 = self._create_named_role(prohibited_name1)
        url = '/roles/{prior_role_id}/implies/{implied_role_id}'.format(prior_role_id=prior_role['id'], implied_role_id=prohibited_role1['id'])
        self.put(url, expected_status=http.client.FORBIDDEN)
        prohibited_role2 = self._create_named_role(prohibited_name2)
        url = '/roles/{prior_role_id}/implies/{implied_role_id}'.format(prior_role_id=prior_role['id'], implied_role_id=prohibited_role2['id'])
        self.put(url, expected_status=http.client.FORBIDDEN)
        accepted_role1 = self._create_named_role(accepted_name1)
        url = '/roles/{prior_role_id}/implies/{implied_role_id}'.format(prior_role_id=prior_role['id'], implied_role_id=accepted_role1['id'])
        self.put(url, expected_status=http.client.CREATED)

    def test_trusts_from_implied_role(self):
        self._create_three_roles()
        self._create_implied_role(self.role_list[0], self.role_list[1])
        self._create_implied_role(self.role_list[1], self.role_list[2])
        self._assign_top_role_to_user_on_project(self.user, self.project)
        trustee = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        ref = unit.new_trust_ref(trustor_user_id=self.user['id'], trustee_user_id=trustee['id'], project_id=self.project['id'], role_ids=[self.role_list[0]['id']])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = r.result['trust']
        self.assertEqual(self.role_list[0]['id'], trust['roles'][0]['id'])
        self.assertThat(trust['roles'], matchers.HasLength(1))
        auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'], trust_id=trust['id'])
        r = self.v3_create_token(auth_data)
        token = r.result['token']
        self.assertThat(token['roles'], matchers.HasLength(len(self.role_list)))
        for role in token['roles']:
            self.assertIn(role, self.role_list)
        for role in self.role_list:
            self.assertIn(role, token['roles'])

    def test_trusts_from_domain_specific_implied_role(self):
        self._create_three_roles()
        role = unit.new_role_ref(domain_id=self.domain_id)
        self.role_list[0] = PROVIDERS.role_api.create_role(role['id'], role)
        self._create_implied_role(self.role_list[0], self.role_list[1])
        self._create_implied_role(self.role_list[1], self.role_list[2])
        self._assign_top_role_to_user_on_project(self.user, self.project)
        trustee = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        ref = unit.new_trust_ref(trustor_user_id=self.user['id'], trustee_user_id=trustee['id'], project_id=self.project['id'], role_ids=[self.role_list[0]['id']])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = r.result['trust']
        self.assertEqual(self.role_list[0]['id'], trust['roles'][0]['id'])
        self.assertThat(trust['roles'], matchers.HasLength(1))
        auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'], trust_id=trust['id'])
        r = self.v3_create_token(auth_data)
        token = r.result['token']
        self.assertThat(token['roles'], matchers.HasLength(len(self.role_list) - 1))
        for role in token['roles']:
            self.assertIn(role, self.role_list)
        for role in [self.role_list[1], self.role_list[2]]:
            self.assertIn(role, token['roles'])
        self.assertNotIn(self.role_list[0], token['roles'])

    def test_global_role_cannot_imply_domain_specific_role(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        domain_role_ref = unit.new_role_ref(domain_id=domain['id'])
        domain_role = PROVIDERS.role_api.create_role(domain_role_ref['id'], domain_role_ref)
        global_role_ref = unit.new_role_ref()
        global_role = PROVIDERS.role_api.create_role(global_role_ref['id'], global_role_ref)
        self.put('/roles/%s/implies/%s' % (global_role['id'], domain_role['id']), expected_status=http.client.FORBIDDEN)