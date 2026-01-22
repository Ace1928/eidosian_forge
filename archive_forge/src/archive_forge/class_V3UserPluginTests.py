import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
class V3UserPluginTests(BaseUserPluginTests, base.BaseAuthTokenTestCase):

    def setUp(self):
        super(V3UserPluginTests, self).setUp()
        self.service_token_id = uuid.uuid4().hex
        self.service_token = fixture.V3Token()
        s = self.service_token.add_service('identity', name='keystone')
        s.add_standard_endpoints(public=BASE_URI, admin=BASE_URI, internal=BASE_URI)
        self.configure_middleware(auth_type='v3password', auth_url='%s/v3/' % AUTH_URL, user_id=self.service_token.user_id, password=uuid.uuid4().hex, project_id=self.service_token.project_id)
        auth_discovery = fixture.DiscoveryList(href=AUTH_URL)
        self.requests_mock.get(AUTH_URL, json=auth_discovery)
        base_discovery = fixture.DiscoveryList(href=BASE_URI)
        self.requests_mock.get(BASE_URI, json=base_discovery)
        self.requests_mock.post('%s/v3/auth/tokens' % AUTH_URL, headers={'X-Subject-Token': self.service_token_id}, json=self.service_token)

    def get_role_names(self, token):
        return [x['name'] for x in token['token'].get('roles', [])]

    def get_token(self, project=True, service=False):
        token_id = uuid.uuid4().hex
        token = fixture.V3Token()
        if project:
            token.set_project_scope()
        token.add_role()
        if service:
            token.add_role('service')
        request_headers = {'X-Auth-Token': self.service_token_id, 'X-Subject-Token': token_id}
        headers = {'X-Subject-Token': token_id}
        self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, request_headers=request_headers, headers=headers, json=token)
        return (token_id, token)

    def assertTokenDataEqual(self, token_id, token, token_data):
        super(V3UserPluginTests, self).assertTokenDataEqual(token_id, token, token_data)
        self.assertEqual(token.user_domain_id, token_data.user_domain_id)
        self.assertEqual(token.project_id, token_data.project_id)
        self.assertEqual(token.project_domain_id, token_data.project_domain_id)

    def test_domain_scope(self):
        token_id, token = self.get_token(project=False)
        token.set_domain_scope()
        plugin = self.get_plugin(token_id)
        self.assertEqual(token.domain_id, plugin.user.domain_id)
        self.assertIsNone(plugin.user.project_id)

    def test_trust_scope(self):
        token_id, token = self.get_token(project=False)
        token.set_trust_scope()
        plugin = self.get_plugin(token_id)
        self.assertEqual(token.trust_id, plugin.user.trust_id)
        self.assertEqual(token.trustor_user_id, plugin.user.trustor_user_id)
        self.assertEqual(token.trustee_user_id, plugin.user.trustee_user_id)