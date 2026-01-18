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
def test_send_authn_req_to_idp(self):
    self.requests_mock.post(self.IDENTITY_PROVIDER_URL, content=saml2_fixtures.SAML2_ASSERTION)
    self.saml2plugin.sp_response_consumer_url = self.SHIB_CONSUMER_URL
    self.saml2plugin.saml2_authn_request = etree.XML(saml2_fixtures.SP_SOAP_RESPONSE)
    self.saml2plugin._send_idp_saml2_authn_request(self.session)
    idp_response = make_oneline(etree.tostring(self.saml2plugin.saml2_idp_authn_response))
    saml2_assertion_oneline = make_oneline(saml2_fixtures.SAML2_ASSERTION)
    error = 'Expected %s instead of %s' % (saml2_fixtures.SAML2_ASSERTION, idp_response)
    self.assertEqual(idp_response, saml2_assertion_oneline, error)