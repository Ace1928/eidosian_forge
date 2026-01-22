import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
class SecurityRequirementsTestCase(test_v3.RestfulTestCase):

    def setUp(self):
        super(SecurityRequirementsTestCase, self).setUp()
        self.non_admin_user = unit.create_user(PROVIDERS.identity_api, CONF.identity.default_domain_id)
        self.admin_user = unit.create_user(PROVIDERS.identity_api, CONF.identity.default_domain_id)
        self.project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(self.project['id'], self.project)
        self.non_admin_role = unit.new_role_ref(name='not_admin')
        PROVIDERS.role_api.create_role(self.non_admin_role['id'], self.non_admin_role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.non_admin_user['id'], self.project['id'], self.role['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.admin_user['id'], self.project['id'], self.role_id)

    def _get_non_admin_token(self):
        non_admin_auth_data = self.build_authentication_request(user_id=self.non_admin_user['id'], password=self.non_admin_user['password'], project_id=self.project['id'])
        return self.get_requested_token(non_admin_auth_data)

    def _get_admin_token(self):
        non_admin_auth_data = self.build_authentication_request(user_id=self.admin_user['id'], password=self.admin_user['password'], project_id=self.project['id'])
        return self.get_requested_token(non_admin_auth_data)

    def test_get_head_security_compliance_config_for_default_domain(self):
        """Ask for all security compliance configuration options.

        Support for enforcing security compliance per domain currently doesn't
        exist. Make sure when we ask for security compliance information, it's
        only for the default domain and that it only returns whitelisted
        options.
        """
        password_regex = uuid.uuid4().hex
        password_regex_description = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex=password_regex)
        self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
        expected_response = {'security_compliance': {'password_regex': password_regex, 'password_regex_description': password_regex_description}}
        url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance'}
        regular_response = self.get(url, token=self._get_non_admin_token())
        self.assertEqual(regular_response.result['config'], expected_response)
        admin_response = self.get(url, token=self._get_admin_token())
        self.assertEqual(admin_response.result['config'], expected_response)
        self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
        self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)

    def test_get_security_compliance_config_for_non_default_domain_fails(self):
        """Getting security compliance opts for other domains should fail.

        Support for enforcing security compliance rules per domain currently
        does not exist, so exposing security compliance information for any
        domain other than the default domain should not be allowed.
        """
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        password_regex = uuid.uuid4().hex
        password_regex_description = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex=password_regex)
        self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
        url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': domain['id'], 'group': 'security_compliance'}
        self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())
        self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_get_non_whitelisted_security_compliance_opt_fails(self):
        """We only support exposing a subset of security compliance options.

        Given that security compliance information is sensitive in nature, we
        should make sure that only the options we want to expose are readable
        via the API.
        """
        self.config_fixture.config(group='security_compliance', lockout_failure_attempts=1)
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance', 'option': 'lockout_failure_attempts'}
        self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())
        self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_get_security_compliance_password_regex(self):
        """Ask for the security compliance password regular expression."""
        password_regex = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex=password_regex)
        group = 'security_compliance'
        option = 'password_regex'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        regular_response = self.get(url, token=self._get_non_admin_token())
        self.assertEqual(regular_response.result['config'][option], password_regex)
        admin_response = self.get(url, token=self._get_admin_token())
        self.assertEqual(admin_response.result['config'][option], password_regex)
        self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
        self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)

    def test_get_security_compliance_password_regex_description(self):
        """Ask for the security compliance password regex description."""
        password_regex_description = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
        group = 'security_compliance'
        option = 'password_regex_description'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        regular_response = self.get(url, token=self._get_non_admin_token())
        self.assertEqual(regular_response.result['config'][option], password_regex_description)
        admin_response = self.get(url, token=self._get_admin_token())
        self.assertEqual(admin_response.result['config'][option], password_regex_description)
        self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
        self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)

    def test_get_security_compliance_password_regex_returns_none(self):
        """When an option isn't set, we should explicitly return None."""
        group = 'security_compliance'
        option = 'password_regex'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        regular_response = self.get(url, token=self._get_non_admin_token())
        self.assertIsNone(regular_response.result['config'][option])
        admin_response = self.get(url, token=self._get_admin_token())
        self.assertIsNone(admin_response.result['config'][option])
        self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
        self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)

    def test_get_security_compliance_password_regex_desc_returns_none(self):
        """When an option isn't set, we should explicitly return None."""
        group = 'security_compliance'
        option = 'password_regex_description'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        regular_response = self.get(url, token=self._get_non_admin_token())
        self.assertIsNone(regular_response.result['config'][option])
        admin_response = self.get(url, token=self._get_admin_token())
        self.assertIsNone(admin_response.result['config'][option])
        self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
        self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)

    def test_get_security_compliance_config_with_user_from_other_domain(self):
        """Make sure users from other domains can access password requirements.

        Even though a user is in a separate domain, they should be able to see
        the security requirements for the deployment. This is because security
        compliance is not yet implemented on a per domain basis. Once that
        happens, then this should no longer be possible since a user should
        only care about the security compliance requirements for the domain
        that they are in.
        """
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.create_user(PROVIDERS.identity_api, domain['id'])
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], project['id'], self.non_admin_role['id'])
        password_regex = uuid.uuid4().hex
        password_regex_description = uuid.uuid4().hex
        group = 'security_compliance'
        self.config_fixture.config(group=group, password_regex=password_regex)
        self.config_fixture.config(group=group, password_regex_description=password_regex_description)
        user_token = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=project['id'])
        user_token = self.get_requested_token(user_token)
        url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group}
        response = self.get(url, token=user_token)
        self.assertEqual(response.result['config'][group]['password_regex'], password_regex)
        self.assertEqual(response.result['config'][group]['password_regex_description'], password_regex_description)
        self.head(url, token=user_token, expected_status=http.client.OK)

    def test_update_security_compliance_config_group_fails(self):
        """Make sure that updates to the entire security group section fail.

        We should only allow the ability to modify a deployments security
        compliance rules through configuration. Especially since it's only
        enforced on the default domain.
        """
        new_config = {'security_compliance': {'password_regex': uuid.uuid4().hex, 'password_regex_description': uuid.uuid4().hex}}
        url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance'}
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_update_security_compliance_password_regex_fails(self):
        """Make sure any updates to security compliance options fail."""
        group = 'security_compliance'
        option = 'password_regex'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        new_config = {group: {option: uuid.uuid4().hex}}
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_update_security_compliance_password_regex_description_fails(self):
        """Make sure any updates to security compliance options fail."""
        group = 'security_compliance'
        option = 'password_regex_description'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        new_config = {group: {option: uuid.uuid4().hex}}
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_update_non_whitelisted_security_compliance_option_fails(self):
        """Updating security compliance options through the API is not allowed.

        Requests to update anything in the security compliance group through
        the API should be Forbidden. This ensures that we are covering cases
        where the option being updated isn't in the white list.
        """
        group = 'security_compliance'
        option = 'lockout_failure_attempts'
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
        new_config = {group: {option: 1}}
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_delete_security_compliance_group_fails(self):
        """The security compliance group shouldn't be deleteable."""
        url = '/domains/%(domain_id)s/config/%(group)s/' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance'}
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_delete_security_compliance_password_regex_fails(self):
        """The security compliance options shouldn't be deleteable."""
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance', 'option': 'password_regex'}
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_delete_security_compliance_password_regex_description_fails(self):
        """The security compliance options shouldn't be deleteable."""
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance', 'option': 'password_regex_description'}
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())

    def test_delete_non_whitelisted_security_compliance_options_fails(self):
        """The security compliance options shouldn't be deleteable."""
        url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance', 'option': 'lockout_failure_attempts'}
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
        self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())