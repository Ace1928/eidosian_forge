import base64
import uuid
import requests
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1 import fixture as ksa_fixtures
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_workflow(self):
    token_id = uuid.uuid4().hex
    token = ksa_fixtures.V3Token()
    self.requests_mock.get(self.default_sp_url, response_list=[dict(headers=CONTENT_TYPE_PAOS_HEADER, content=utils.make_oneline(saml2_fixtures.SP_SOAP_RESPONSE)), dict(headers={'X-Subject-Token': token_id}, json=token)])
    authm = self.requests_mock.post(self.TEST_IDP_URL, content=saml2_fixtures.SAML2_ASSERTION)
    self.requests_mock.post(self.TEST_CONSUMER_URL, status_code=302, headers={'Location': self.sp_url()})
    auth_ref = self.get_plugin().get_auth_ref(self.session)
    self.assertEqual(token_id, auth_ref.auth_token)
    self.assertEqual(self.calls, [self.default_sp_url, self.TEST_IDP_URL, self.TEST_CONSUMER_URL, self.default_sp_url])
    self.assertEqual(self.basic_header(), authm.last_request.headers['Authorization'])
    authn_request = self.requests_mock.request_history[1].text
    self.assertThat(saml2_fixtures.AUTHN_REQUEST, matchers.XMLEquals(authn_request))