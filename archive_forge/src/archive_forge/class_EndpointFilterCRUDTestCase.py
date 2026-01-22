import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
class EndpointFilterCRUDTestCase(EndpointFilterTestCase):

    def test_create_endpoint_project_association(self):
        """PUT /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Valid endpoint and project id test case.

        """
        self.put(self.default_request_url)

    def test_create_endpoint_project_association_with_invalid_project(self):
        """PUT OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid project id test case.

        """
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': uuid.uuid4().hex, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NOT_FOUND)

    def test_create_endpoint_project_association_with_invalid_endpoint(self):
        """PUT /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid endpoint id test case.

        """
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)

    def test_create_endpoint_project_association_with_unexpected_body(self):
        """PUT /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Unexpected body in request. The body should be ignored.

        """
        self.put(self.default_request_url, body={'project_id': self.default_domain_project_id})

    def test_check_endpoint_project_association(self):
        """HEAD /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Valid project and endpoint id test case.

        """
        self.put(self.default_request_url)
        self.head('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NO_CONTENT)

    def test_check_endpoint_project_association_with_invalid_project(self):
        """HEAD /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid project id test case.

        """
        self.put(self.default_request_url)
        self.head('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': uuid.uuid4().hex, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NOT_FOUND)

    def test_check_endpoint_project_association_with_invalid_endpoint(self):
        """HEAD /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid endpoint id test case.

        """
        self.put(self.default_request_url)
        self.head('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)

    def test_get_endpoint_project_association(self):
        """GET /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Valid project and endpoint id test case.

        """
        self.put(self.default_request_url)
        self.get('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NO_CONTENT)

    def test_get_endpoint_project_association_with_invalid_project(self):
        """GET /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid project id test case.

        """
        self.put(self.default_request_url)
        self.get('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': uuid.uuid4().hex, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NOT_FOUND)

    def test_get_endpoint_project_association_with_invalid_endpoint(self):
        """GET /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid endpoint id test case.

        """
        self.put(self.default_request_url)
        self.get('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)

    def test_list_endpoints_associated_with_valid_project(self):
        """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoints.

        Valid project and endpoint id test case.

        """
        self.put(self.default_request_url)
        resource_url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints' % {'project_id': self.default_domain_project_id}
        r = self.get(resource_url)
        self.assertValidEndpointListResponse(r, self.endpoint, resource_url=resource_url)
        self.head(resource_url, expected_status=http.client.OK)

    def test_list_endpoints_associated_with_invalid_project(self):
        """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoints.

        Invalid project id test case.

        """
        self.put(self.default_request_url)
        url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints' % {'project_id': uuid.uuid4().hex}
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_list_projects_associated_with_endpoint(self):
        """GET & HEAD /OS-EP-FILTER/endpoints/{endpoint_id}/projects.

        Valid endpoint-project association test case.

        """
        self.put(self.default_request_url)
        resource_url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': self.endpoint_id}
        r = self.get(resource_url, expected_status=http.client.OK)
        self.assertValidProjectListResponse(r, self.default_domain_project, resource_url=resource_url)
        self.head(resource_url, expected_status=http.client.OK)

    def test_list_projects_with_no_endpoint_project_association(self):
        """GET & HEAD /OS-EP-FILTER/endpoints/{endpoint_id}/projects.

        Valid endpoint id but no endpoint-project associations test case.

        """
        url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': self.endpoint_id}
        r = self.get(url, expected_status=http.client.OK)
        self.assertValidProjectListResponse(r, expected_length=0)
        self.head(url, expected_status=http.client.OK)

    def test_list_projects_associated_with_invalid_endpoint(self):
        """GET & HEAD /OS-EP-FILTER/endpoints/{endpoint_id}/projects.

        Invalid endpoint id test case.

        """
        url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': uuid.uuid4().hex}
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_remove_endpoint_project_association(self):
        """DELETE /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Valid project id and endpoint id test case.

        """
        self.put(self.default_request_url)
        self.delete('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': self.endpoint_id})

    def test_remove_endpoint_project_association_with_invalid_project(self):
        """DELETE /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid project id test case.

        """
        self.put(self.default_request_url)
        self.delete('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': uuid.uuid4().hex, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NOT_FOUND)

    def test_remove_endpoint_project_association_with_invalid_endpoint(self):
        """DELETE /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Invalid endpoint id test case.

        """
        self.put(self.default_request_url)
        self.delete('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)

    def test_endpoint_project_association_cleanup_when_project_deleted(self):
        self.put(self.default_request_url)
        association_url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': self.endpoint_id}
        r = self.get(association_url)
        self.assertValidProjectListResponse(r, expected_length=1)
        self.delete('/projects/%(project_id)s' % {'project_id': self.default_domain_project_id})
        r = self.get(association_url)
        self.assertValidProjectListResponse(r, expected_length=0)

    def test_endpoint_project_association_cleanup_when_endpoint_deleted(self):
        self.put(self.default_request_url)
        association_url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints' % {'project_id': self.default_domain_project_id}
        r = self.get(association_url)
        self.assertValidEndpointListResponse(r, expected_length=1)
        self.delete('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id})
        r = self.get(association_url)
        self.assertValidEndpointListResponse(r, expected_length=0)

    @unit.skip_if_cache_disabled('catalog')
    def test_create_endpoint_project_association_invalidates_cache(self):
        endpoint_id2 = uuid.uuid4().hex
        endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='public', id=endpoint_id2)
        PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2.copy())
        self.put(self.default_request_url)
        user_id = uuid.uuid4().hex
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(1, len(catalog[0]['endpoints']))
        self.assertEqual(self.endpoint_id, catalog[0]['endpoints'][0]['id'])
        PROVIDERS.catalog_api.driver.add_endpoint_to_project(endpoint_id2, self.default_domain_project_id)
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(1, len(catalog[0]['endpoints']))
        PROVIDERS.catalog_api.driver.remove_endpoint_from_project(endpoint_id2, self.default_domain_project_id)
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': endpoint_id2})
        catalog = self.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(2, len(catalog[0]['endpoints']))
        ep_id_list = [catalog[0]['endpoints'][0]['id'], catalog[0]['endpoints'][1]['id']]
        self.assertCountEqual([self.endpoint_id, endpoint_id2], ep_id_list)

    @unit.skip_if_cache_disabled('catalog')
    def test_remove_endpoint_from_project_invalidates_cache(self):
        endpoint_id2 = uuid.uuid4().hex
        endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='public', id=endpoint_id2)
        PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2.copy())
        self.put(self.default_request_url)
        self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': endpoint_id2})
        user_id = uuid.uuid4().hex
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        ep_id_list = [catalog[0]['endpoints'][0]['id'], catalog[0]['endpoints'][1]['id']]
        self.assertEqual(2, len(catalog[0]['endpoints']))
        self.assertCountEqual([self.endpoint_id, endpoint_id2], ep_id_list)
        PROVIDERS.catalog_api.driver.remove_endpoint_from_project(endpoint_id2, self.default_domain_project_id)
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(2, len(catalog[0]['endpoints']))
        PROVIDERS.catalog_api.driver.add_endpoint_to_project(endpoint_id2, self.default_domain_project_id)
        self.delete('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': endpoint_id2})
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
        self.assertEqual(1, len(catalog[0]['endpoints']))
        self.assertEqual(self.endpoint_id, catalog[0]['endpoints'][0]['id'])