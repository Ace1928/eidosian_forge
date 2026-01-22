import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class IdentityTestFilteredCase(filtering.FilterTests, test_v3.RestfulTestCase):
    """Test filter enforcement on the v3 Identity API."""

    def _policy_fixture(self):
        return ksfixtures.Policy(self.config_fixture, policy_file=self.tmpfilename)

    def setUp(self):
        """Setup for Identity Filter Test Cases."""
        self.tempfile = self.useFixture(temporaryfile.SecureTempFile())
        self.tmpfilename = self.tempfile.file_name
        super(IdentityTestFilteredCase, self).setUp()

    def load_sample_data(self):
        """Create sample data for these tests.

        As well as the usual housekeeping, create a set of domains,
        users, roles and projects for the subsequent tests:

        - Three domains: A,B & C.  C is disabled.
        - DomainA has user1, DomainB has user2 and user3
        - DomainA has group1 and group2, DomainB has group3
        - User1 has a role on DomainA

        Remember that there will also be a fourth domain in existence,
        the default domain.

        """
        self._populate_default_domain()
        self.domainA = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainA['id'], self.domainA)
        self.domainB = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainB['id'], self.domainB)
        self.domainC = unit.new_domain_ref()
        self.domainC['enabled'] = False
        PROVIDERS.resource_api.create_domain(self.domainC['id'], self.domainC)
        self.user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainA['id'])
        self.user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        self.user3 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        self.role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(self.role['id'], self.role)
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user1['id'], domain_id=self.domainA['id'])
        self.auth = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'])

    def _get_id_list_from_ref_list(self, ref_list):
        result_list = []
        for x in ref_list:
            result_list.append(x['id'])
        return result_list

    def _set_policy(self, new_policy):
        with open(self.tmpfilename, 'w') as policyfile:
            policyfile.write(jsonutils.dumps(new_policy))

    def test_list_users_filtered_by_domain(self):
        """GET /users?domain_id=mydomain (filtered).

        Test Plan:

        - Update policy so api is unprotected
        - Use an un-scoped token to make sure we can filter the
          users by domainB, getting back the 2 users in that domain

        """
        self._set_policy({'identity:list_users': []})
        url_by_name = '/users?domain_id=%s' % self.domainB['id']
        r = self.get(url_by_name, auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('users'))
        self.assertIn(self.user2['id'], id_list)
        self.assertIn(self.user3['id'], id_list)

    def test_list_filtered_domains(self):
        """GET /domains?enabled=0.

        Test Plan:

        - Update policy for no protection on api
        - Filter by the 'enabled' boolean to get disabled domains, which
          should return just domainC
        - Try the filter using different ways of specifying True/False
          to test that our handling of booleans in filter matching is
          correct

        """
        new_policy = {'identity:list_domains': []}
        self._set_policy(new_policy)
        r = self.get('/domains?enabled=0', auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
        self.assertEqual(1, len(id_list))
        self.assertIn(self.domainC['id'], id_list)
        for val in ('0', 'false', 'False', 'FALSE', 'n', 'no', 'off'):
            r = self.get('/domains?enabled=%s' % val, auth=self.auth)
            id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
            self.assertEqual([self.domainC['id']], id_list)
        for val in ('1', 'true', 'True', 'TRUE', 'y', 'yes', 'on'):
            r = self.get('/domains?enabled=%s' % val, auth=self.auth)
            id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
            self.assertEqual(3, len(id_list))
            self.assertIn(self.domainA['id'], id_list)
            self.assertIn(self.domainB['id'], id_list)
            self.assertIn(CONF.identity.default_domain_id, id_list)
        r = self.get('/domains?enabled', auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
        self.assertEqual(3, len(id_list))
        self.assertIn(self.domainA['id'], id_list)
        self.assertIn(self.domainB['id'], id_list)
        self.assertIn(CONF.identity.default_domain_id, id_list)

    def test_multiple_filters(self):
        """GET /domains?enabled&name=myname.

        Test Plan:

        - Update policy for no protection on api
        - Filter by the 'enabled' boolean and name - this should
          return a single domain

        """
        new_policy = {'identity:list_domains': []}
        self._set_policy(new_policy)
        my_url = '/domains?enabled&name=%s' % self.domainA['name']
        r = self.get(my_url, auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
        self.assertEqual(1, len(id_list))
        self.assertIn(self.domainA['id'], id_list)
        self.assertIs(True, r.result.get('domains')[0]['enabled'])

    def test_invalid_filter_is_ignored(self):
        """GET /domains?enableds&name=myname.

        Test Plan:

        - Update policy for no protection on api
        - Filter by name and 'enableds', which does not exist
        - Assert 'enableds' is ignored

        """
        new_policy = {'identity:list_domains': []}
        self._set_policy(new_policy)
        my_url = '/domains?enableds=0&name=%s' % self.domainA['name']
        r = self.get(my_url, auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
        self.assertEqual(1, len(id_list))
        self.assertIn(self.domainA['id'], id_list)
        self.assertIs(True, r.result.get('domains')[0]['enabled'])

    def test_list_users_filtered_by_funny_name(self):
        """GET /users?name=%myname%.

        Test Plan:

        - Update policy so api is unprotected
        - Update a user with name that has filter escape characters
        - Ensure we can filter on it

        """
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            self._set_policy({'identity:list_users': []})
            user = self.user1
            user['name'] = '%my%name%'
            PROVIDERS.identity_api.update_user(user['id'], user)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            url_by_name = '/users?name=%my%name%'
            r = self.get(url_by_name, auth=self.auth)
            self.assertEqual(1, len(r.result.get('users')))
            self.assertEqual(user['id'], r.result.get('users')[0]['id'])

    def test_inexact_filters(self):
        user_list = self._create_test_data('user', 20)
        user = user_list[5]
        user['name'] = 'The'
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = user_list[6]
        user['name'] = 'The Ministry'
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = user_list[7]
        user['name'] = 'The Ministry of'
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = user_list[8]
        user['name'] = 'The Ministry of Silly'
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = user_list[9]
        user['name'] = 'The Ministry of Silly Walks'
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = user_list[10]
        user['name'] = 'the ministry of silly walks OF'
        PROVIDERS.identity_api.update_user(user['id'], user)
        self._set_policy({'identity:list_users': []})
        url_by_name = '/users?name__contains=Ministry'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(4, len(r.result.get('users')))
        self._match_with_list(r.result.get('users'), user_list, list_start=6, list_end=10)
        url_by_name = '/users?name__icontains=miNIstry'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(5, len(r.result.get('users')))
        self._match_with_list(r.result.get('users'), user_list, list_start=6, list_end=11)
        url_by_name = '/users?name__startswith=The'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(5, len(r.result.get('users')))
        self._match_with_list(r.result.get('users'), user_list, list_start=5, list_end=10)
        url_by_name = '/users?name__istartswith=the'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(6, len(r.result.get('users')))
        self._match_with_list(r.result.get('users'), user_list, list_start=5, list_end=11)
        url_by_name = '/users?name__endswith=of'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(1, len(r.result.get('users')))
        self.assertEqual(user_list[7]['id'], r.result.get('users')[0]['id'])
        url_by_name = '/users?name__iendswith=OF'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(2, len(r.result.get('users')))
        self.assertEqual(user_list[7]['id'], r.result.get('users')[0]['id'])
        self.assertEqual(user_list[10]['id'], r.result.get('users')[1]['id'])
        self._delete_test_data('user', user_list)

    def test_filter_sql_injection_attack(self):
        """GET /users?name=<injected sql_statement>.

        Test Plan:

        - Attempt to get all entities back by passing a two-term attribute
        - Attempt to piggyback filter to damage DB (e.g. drop table)

        """
        self._set_policy({'identity:list_users': [], 'identity:list_groups': [], 'identity:create_group': []})
        url_by_name = "/users?name=anything' or 'x'='x"
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(0, len(r.result.get('users')))
        group = unit.new_group_ref(domain_id=self.domainB['id'])
        group = PROVIDERS.identity_api.create_group(group)
        url_by_name = "/users?name=x'; drop table group"
        r = self.get(url_by_name, auth=self.auth)
        url_by_name = '/groups'
        r = self.get(url_by_name, auth=self.auth)
        self.assertGreater(len(r.result.get('groups')), 0)