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
class TestAuthSpecificData(test_v3.RestfulTestCase):

    def test_get_catalog_with_project_scoped_token(self):
        """Call ``GET /auth/catalog`` with a project-scoped token."""
        r = self.get('/auth/catalog', expected_status=http.client.OK)
        self.assertValidCatalogResponse(r)

    def test_head_catalog_with_project_scoped_token(self):
        """Call ``HEAD /auth/catalog`` with a project-scoped token."""
        self.head('/auth/catalog', expected_status=http.client.OK)

    def test_get_catalog_with_domain_scoped_token(self):
        """Call ``GET /auth/catalog`` with a domain-scoped token."""
        self.put(path='/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id']))
        self.get('/auth/catalog', auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain['id']), expected_status=http.client.FORBIDDEN)

    def test_head_catalog_with_domain_scoped_token(self):
        """Call ``HEAD /auth/catalog`` with a domain-scoped token."""
        self.put(path='/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id']))
        self.head('/auth/catalog', auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain['id']), expected_status=http.client.FORBIDDEN)

    def test_get_catalog_with_unscoped_token(self):
        """Call ``GET /auth/catalog`` with an unscoped token."""
        self.get('/auth/catalog', auth=self.build_authentication_request(user_id=self.default_domain_user['id'], password=self.default_domain_user['password']), expected_status=http.client.FORBIDDEN)

    def test_head_catalog_with_unscoped_token(self):
        """Call ``HEAD /auth/catalog`` with an unscoped token."""
        self.head('/auth/catalog', auth=self.build_authentication_request(user_id=self.default_domain_user['id'], password=self.default_domain_user['password']), expected_status=http.client.FORBIDDEN)

    def test_get_catalog_no_token(self):
        """Call ``GET /auth/catalog`` without a token."""
        self.get('/auth/catalog', noauth=True, expected_status=http.client.UNAUTHORIZED)

    def test_head_catalog_no_token(self):
        """Call ``HEAD /auth/catalog`` without a token."""
        self.head('/auth/catalog', noauth=True, expected_status=http.client.UNAUTHORIZED)

    def test_get_projects_with_project_scoped_token(self):
        r = self.get('/auth/projects', expected_status=http.client.OK)
        self.assertThat(r.json['projects'], matchers.HasLength(1))
        self.assertValidProjectListResponse(r)

    def test_head_projects_with_project_scoped_token(self):
        self.head('/auth/projects', expected_status=http.client.OK)

    def test_get_projects_matches_federated_get_projects(self):
        ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        r = self.post('/projects', body={'project': ref})
        unauthorized_project_id = r.json['project']['id']
        r = self.get('/auth/projects', expected_status=http.client.OK)
        self.assertThat(r.json['projects'], matchers.HasLength(1))
        for project in r.json['projects']:
            self.assertNotEqual(unauthorized_project_id, project['id'])
        expected_project_id = r.json['projects'][0]['id']
        r = self.get('/OS-FEDERATION/projects', expected_status=http.client.OK)
        self.assertThat(r.json['projects'], matchers.HasLength(1))
        for project in r.json['projects']:
            self.assertEqual(expected_project_id, project['id'])

    def test_get_domains_matches_federated_get_domains(self):
        ref = unit.new_domain_ref()
        r = self.post('/domains', body={'domain': ref})
        unauthorized_domain_id = r.json['domain']['id']
        ref = unit.new_domain_ref()
        r = self.post('/domains', body={'domain': ref})
        authorized_domain_id = r.json['domain']['id']
        path = '/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s' % {'domain_id': authorized_domain_id, 'user_id': self.user_id, 'role_id': self.role_id}
        self.put(path, expected_status=http.client.NO_CONTENT)
        r = self.get('/auth/domains', expected_status=http.client.OK)
        self.assertThat(r.json['domains'], matchers.HasLength(1))
        self.assertEqual(authorized_domain_id, r.json['domains'][0]['id'])
        self.assertNotEqual(unauthorized_domain_id, r.json['domains'][0]['id'])
        r = self.get('/OS-FEDERATION/domains', expected_status=http.client.OK)
        self.assertThat(r.json['domains'], matchers.HasLength(1))
        self.assertEqual(authorized_domain_id, r.json['domains'][0]['id'])
        self.assertNotEqual(unauthorized_domain_id, r.json['domains'][0]['id'])

    def test_get_domains_with_project_scoped_token(self):
        self.put(path='/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id']))
        r = self.get('/auth/domains', expected_status=http.client.OK)
        self.assertThat(r.json['domains'], matchers.HasLength(1))
        self.assertValidDomainListResponse(r)

    def test_head_domains_with_project_scoped_token(self):
        self.put(path='/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id']))
        self.head('/auth/domains', expected_status=http.client.OK)

    def test_get_system_roles_with_unscoped_token(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        unscoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.post('/auth/tokens', body=unscoped_request)
        unscoped_token = r.headers.get('X-Subject-Token')
        self.assertValidUnscopedTokenResponse(r)
        response = self.get('/auth/system', token=unscoped_token)
        self.assertTrue(response.json_body['system'][0]['all'])
        self.head('/auth/system', token=unscoped_token, expected_status=http.client.OK)

    def test_get_system_roles_returns_empty_list_without_system_roles(self):
        unscoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.post('/auth/tokens', body=unscoped_request)
        unscoped_token = r.headers.get('X-Subject-Token')
        self.assertValidUnscopedTokenResponse(r)
        response = self.get('/auth/system', token=unscoped_token)
        self.assertEqual(response.json_body['system'], [])
        self.head('/auth/system', token=unscoped_token, expected_status=http.client.OK)
        project_scoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_id)
        r = self.post('/auth/tokens', body=project_scoped_request)
        project_scoped_token = r.headers.get('X-Subject-Token')
        self.assertValidProjectScopedTokenResponse(r)
        response = self.get('/auth/system', token=project_scoped_token)
        self.assertEqual(response.json_body['system'], [])
        self.head('/auth/system', token=project_scoped_token, expected_status=http.client.OK)

    def test_get_system_roles_with_project_scoped_token(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        self.put(path='/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id']))
        domain_scoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain['id'])
        r = self.post('/auth/tokens', body=domain_scoped_request)
        domain_scoped_token = r.headers.get('X-Subject-Token')
        self.assertValidDomainScopedTokenResponse(r)
        response = self.get('/auth/system', token=domain_scoped_token)
        self.assertTrue(response.json_body['system'][0]['all'])
        self.head('/auth/system', token=domain_scoped_token, expected_status=http.client.OK)

    def test_get_system_roles_with_domain_scoped_token(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        project_scoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_id)
        r = self.post('/auth/tokens', body=project_scoped_request)
        project_scoped_token = r.headers.get('X-Subject-Token')
        self.assertValidProjectScopedTokenResponse(r)
        response = self.get('/auth/system', token=project_scoped_token)
        self.assertTrue(response.json_body['system'][0]['all'])
        self.head('/auth/system', token=project_scoped_token, expected_status=http.client.OK)