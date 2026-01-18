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
def test_scope_saml2_token_to_domain(self):
    self.stub_auth(json=self.DOMAIN_SCOPED_TOKEN_JSON)
    token = self.saml2_scope_plugin.get_auth_ref(self.session)
    self.assertTrue(token.domain_scoped, 'Received token is not scoped')
    self.assertEqual(client_fixtures.AUTH_SUBJECT_TOKEN, token.auth_token)
    self.assertEqual(self.TEST_DOMAIN_ID, token.domain_id)
    self.assertEqual(self.TEST_DOMAIN_NAME, token.domain_name)