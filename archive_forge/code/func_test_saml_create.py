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
def test_saml_create(self):
    """Test that a token can be exchanged for a SAML assertion."""
    token_id = uuid.uuid4().hex
    service_provider_id = uuid.uuid4().hex
    self.requests_mock.post(self.SAML2_FULL_URL, text=saml2_fixtures.TOKEN_BASED_SAML)
    text = self.manager.create_saml_assertion(service_provider_id, token_id)
    self.assertEqual(saml2_fixtures.TOKEN_BASED_SAML, text)
    req_json = self.requests_mock.last_request.json()
    self.assertEqual(token_id, req_json['auth']['identity']['token']['id'])
    self.assertEqual(service_provider_id, req_json['auth']['scope']['service_provider']['id'])
    self.assertRequestHeaderEqual('Content-Type', 'application/json')