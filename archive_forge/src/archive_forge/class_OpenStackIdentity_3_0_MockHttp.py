import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
class OpenStackIdentity_3_0_MockHttp(MockHttp):
    fixtures = ComputeFileFixtures('openstack_identity/v3')
    json_content_headers = {'content-type': 'application/json; charset=UTF-8'}

    def _v3(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_versions.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_domains(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_domains.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_projects(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_projects.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_projects_UNAUTHORIZED(self, method, url, body, headers):
        if method == 'GET':
            body = ComputeFileFixtures('openstack').load('_v3__auth.json')
            return (httplib.UNAUTHORIZED, body, self.json_content_headers, httplib.responses[httplib.UNAUTHORIZED])
        raise NotImplementedError()

    def _v3_OS_FEDERATION_identity_providers_test_user_id_protocols_test_tenant_auth(self, method, url, body, headers):
        if method == 'GET':
            if 'Authorization' not in headers:
                return (httplib.UNAUTHORIZED, '', headers, httplib.responses[httplib.OK])
            if headers['Authorization'] == 'Bearer test_key':
                response_body = ComputeFileFixtures('openstack').load('_v3__auth.json')
                response_headers = {'Content-Type': 'application/json', 'x-subject-token': 'foo-bar'}
                return (httplib.OK, response_body, response_headers, httplib.responses[httplib.OK])
            return (httplib.UNAUTHORIZED, '{}', headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_auth_tokens(self, method, url, body, headers):
        if method == 'GET':
            body = json.loads(ComputeFileFixtures('openstack').load('_v3__auth.json'))
            body['token']['expires_at'] = TOMORROW.isoformat()
            headers = self.json_content_headers.copy()
            headers['x-subject-token'] = '00000000000000000000000000000000'
            return (httplib.OK, json.dumps(body), headers, httplib.responses[httplib.OK])
        if method == 'POST':
            status = httplib.OK
            data = json.loads(body)
            if 'password' in data['auth']['identity']:
                if data['auth']['identity']['password']['user']['domain']['name'] != 'test_domain' or data['auth']['scope']['project']['domain']['id'] != 'test_tenant_domain_id':
                    status = httplib.UNAUTHORIZED
            body = ComputeFileFixtures('openstack').load('_v3__auth.json')
            headers = self.json_content_headers.copy()
            headers['x-subject-token'] = '00000000000000000000000000000000'
            return (status, body, headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_auth_tokens_GET_UNAUTHORIZED_POST_OK(self, method, url, body, headers):
        if method == 'GET':
            body = ComputeFileFixtures('openstack').load('_v3__auth_unauthorized.json')
            return (httplib.UNAUTHORIZED, body, self.json_content_headers, httplib.responses[httplib.UNAUTHORIZED])
        if method == 'POST':
            return self._v3_auth_tokens(method, url, body, headers)
        raise NotImplementedError()

    def _v3_users(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_users.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('v3_create_user.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.CREATED])
        raise NotImplementedError()

    def _v3_users_a(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_users_a.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'PATCH':
            body = self.fixtures.load('v3_users_a.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_users_b(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_users_b.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_users_c(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_users_c.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_roles(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_roles.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_domains_default_users_a_roles_a(self, method, url, body, headers):
        if method == 'PUT':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.NO_CONTENT])
        elif method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.NO_CONTENT])
        raise NotImplementedError()

    def _v3_projects_a_users_a_roles_a(self, method, url, body, headers):
        if method == 'PUT':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.NO_CONTENT])
        elif method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.NO_CONTENT])
        raise NotImplementedError()

    def _v3_domains_default(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_domains_default.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_users_a_projects(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_users_a_projects.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_domains_default_users_a_roles(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v3_domains_default_users_a_roles.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_OS_FEDERATION_identity_providers_idp_protocols_oidc_auth(self, method, url, body, headers):
        if method == 'GET':
            headers = self.json_content_headers.copy()
            headers['x-subject-token'] = '00000000000000000000000000000000'
            return (httplib.OK, body, headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_OS_FEDERATION_projects(self, method, url, body, headers):
        if method == 'GET':
            body = json.dumps({'projects': [{'id': 'project_id', 'name': 'project_name'}, {'id': 'project_id2', 'name': 'project_name2'}]})
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v3_auth_projects(self, method, url, body, headers):
        if method == 'GET':
            body = json.dumps({'projects': [{'id': 'project_id', 'name': 'project_name'}, {'id': 'project_id2', 'name': 'project_name2'}]})
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()