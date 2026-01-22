import os
import urllib.parse
import uuid
from lxml import etree
from oslo_config import fixture as config
import requests
from keystoneclient.auth import conf
from keystoneclient.contrib.auth.v3 import saml2
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import saml2_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib.federation import saml as saml_manager
class ScopeFederationTokenTests(AuthenticateviaSAML2Tests):
    TEST_TOKEN = client_fixtures.AUTH_SUBJECT_TOKEN

    def setUp(self):
        super(ScopeFederationTokenTests, self).setUp()
        self.PROJECT_SCOPED_TOKEN_JSON = client_fixtures.project_scoped_token()
        self.PROJECT_SCOPED_TOKEN_JSON['methods'] = ['saml2']
        self.TEST_TENANT_ID = self.PROJECT_SCOPED_TOKEN_JSON.project_id
        self.TEST_TENANT_NAME = self.PROJECT_SCOPED_TOKEN_JSON.project_name
        self.DOMAIN_SCOPED_TOKEN_JSON = client_fixtures.domain_scoped_token()
        self.DOMAIN_SCOPED_TOKEN_JSON['methods'] = ['saml2']
        self.TEST_DOMAIN_ID = self.DOMAIN_SCOPED_TOKEN_JSON.domain_id
        self.TEST_DOMAIN_NAME = self.DOMAIN_SCOPED_TOKEN_JSON.domain_name
        self.saml2_scope_plugin = saml2.Saml2ScopedToken(self.TEST_URL, saml2_fixtures.UNSCOPED_TOKEN_HEADER, project_id=self.TEST_TENANT_ID)

    def test_scope_saml2_token_to_project(self):
        self.stub_auth(json=self.PROJECT_SCOPED_TOKEN_JSON)
        token = self.saml2_scope_plugin.get_auth_ref(self.session)
        self.assertTrue(token.project_scoped, 'Received token is not scoped')
        self.assertEqual(client_fixtures.AUTH_SUBJECT_TOKEN, token.auth_token)
        self.assertEqual(self.TEST_TENANT_ID, token.project_id)
        self.assertEqual(self.TEST_TENANT_NAME, token.project_name)

    def test_scope_saml2_token_to_invalid_project(self):
        self.stub_auth(status_code=401)
        self.saml2_scope_plugin.project_id = uuid.uuid4().hex
        self.saml2_scope_plugin.project_name = None
        self.assertRaises(exceptions.Unauthorized, self.saml2_scope_plugin.get_auth_ref, self.session)

    def test_scope_saml2_token_to_invalid_domain(self):
        self.stub_auth(status_code=401)
        self.saml2_scope_plugin.project_id = None
        self.saml2_scope_plugin.project_name = None
        self.saml2_scope_plugin.domain_id = uuid.uuid4().hex
        self.saml2_scope_plugin.domain_name = None
        self.assertRaises(exceptions.Unauthorized, self.saml2_scope_plugin.get_auth_ref, self.session)

    def test_scope_saml2_token_to_domain(self):
        self.stub_auth(json=self.DOMAIN_SCOPED_TOKEN_JSON)
        token = self.saml2_scope_plugin.get_auth_ref(self.session)
        self.assertTrue(token.domain_scoped, 'Received token is not scoped')
        self.assertEqual(client_fixtures.AUTH_SUBJECT_TOKEN, token.auth_token)
        self.assertEqual(self.TEST_DOMAIN_ID, token.domain_id)
        self.assertEqual(self.TEST_DOMAIN_NAME, token.domain_name)

    def test_dont_set_project_nor_domain(self):
        self.saml2_scope_plugin.project_id = None
        self.saml2_scope_plugin.domain_id = None
        self.assertRaises(exceptions.ValidationError, saml2.Saml2ScopedToken, self.TEST_URL, client_fixtures.AUTH_SUBJECT_TOKEN)