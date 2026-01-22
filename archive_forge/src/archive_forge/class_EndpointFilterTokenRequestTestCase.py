import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
class EndpointFilterTokenRequestTestCase(EndpointFilterTestCase):

    def test_project_scoped_token_using_endpoint_filter(self):
        """Verify endpoints from project scoped token filtered."""
        ref = unit.new_project_ref(domain_id=self.domain_id)
        r = self.post('/projects', body={'project': ref})
        project = self.assertValidProjectResponse(r, ref)
        self.put('/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'project_id': project['id'], 'role_id': self.role['id']})
        body = {'user': {'default_project_id': project['id']}}
        r = self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body=body)
        self.assertValidUserResponse(r)
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': project['id'], 'endpoint_id': self.endpoint_id})
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
        self.assertEqual(project['id'], r.result['token']['project']['id'])

    def test_default_scoped_token_using_endpoint_filter(self):
        """Verify endpoints from default scoped token filtered."""
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
        self.assertEqual(self.project['id'], r.result['token']['project']['id'])
        self.assertIn('name', r.result['token']['catalog'][0])
        endpoint = r.result['token']['catalog'][0]['endpoints'][0]
        self.assertIn('region', endpoint)
        self.assertIn('region_id', endpoint)
        self.assertEqual(endpoint['region'], endpoint['region_id'])

    def test_scoped_token_with_no_catalog_using_endpoint_filter(self):
        """Verify endpoint filter does not affect no catalog."""
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.post('/auth/tokens?nocatalog', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=False)
        self.assertEqual(self.project['id'], r.result['token']['project']['id'])

    def test_invalid_endpoint_project_association(self):
        """Verify an invalid endpoint-project association is handled."""
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
        endpoint_id2 = uuid.uuid4().hex
        endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='public', id=endpoint_id2)
        PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2.copy())
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint_id2})
        PROVIDERS.catalog_api.delete_endpoint(endpoint_id2)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
        self.assertEqual(self.project['id'], r.result['token']['project']['id'])

    def test_disabled_endpoint(self):
        """Test that a disabled endpoint is handled."""
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
        disabled_endpoint_ref = copy.copy(self.endpoint)
        disabled_endpoint_id = uuid.uuid4().hex
        disabled_endpoint_ref.update({'id': disabled_endpoint_id, 'enabled': False, 'interface': 'internal'})
        PROVIDERS.catalog_api.create_endpoint(disabled_endpoint_id, disabled_endpoint_ref)
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': disabled_endpoint_id})
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.post('/auth/tokens', body=auth_data)
        endpoints = r.result['token']['catalog'][0]['endpoints']
        endpoint_ids = [ep['id'] for ep in endpoints]
        self.assertEqual([self.endpoint_id], endpoint_ids)

    def test_multiple_endpoint_project_associations(self):

        def _create_an_endpoint():
            endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
            r = self.post('/endpoints', body={'endpoint': endpoint_ref})
            return r.result['endpoint']['id']
        endpoint_id1 = _create_an_endpoint()
        endpoint_id2 = _create_an_endpoint()
        _create_an_endpoint()
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint_id1})
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint_id2})
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=2)

    def test_get_auth_catalog_using_endpoint_filter(self):
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        token_data = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(token_data, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
        auth_catalog = self.get('/auth/catalog', token=token_data.headers['X-Subject-Token'])
        self.assertEqual(token_data.result['token']['catalog'], auth_catalog.result['catalog'])