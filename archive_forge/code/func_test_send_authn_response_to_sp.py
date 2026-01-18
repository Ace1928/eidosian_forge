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
def test_send_authn_response_to_sp(self):
    self.requests_mock.post(self.SHIB_CONSUMER_URL, json=saml2_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': saml2_fixtures.UNSCOPED_TOKEN_HEADER})
    self.saml2plugin.relay_state = etree.XML(saml2_fixtures.SP_SOAP_RESPONSE).xpath(self.ECP_RELAY_STATE, namespaces=self.ECP_SAML2_NAMESPACES)[0]
    self.saml2plugin.saml2_idp_authn_response = etree.XML(saml2_fixtures.SAML2_ASSERTION)
    self.saml2plugin.idp_response_consumer_url = self.SHIB_CONSUMER_URL
    self.saml2plugin._send_service_provider_saml2_authn_response(self.session)
    token_json = self.saml2plugin.authenticated_response.json()['token']
    token = self.saml2plugin.authenticated_response.headers['X-Subject-Token']
    self.assertEqual(saml2_fixtures.UNSCOPED_TOKEN['token'], token_json)
    self.assertEqual(saml2_fixtures.UNSCOPED_TOKEN_HEADER, token)