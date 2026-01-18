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
def test_standard_workflow_302_redirect(self):
    text = uuid.uuid4().hex
    self.requests_mock.get(self.TEST_SP_URL, response_list=[dict(headers=CONTENT_TYPE_PAOS_HEADER, content=utils.make_oneline(saml2_fixtures.SP_SOAP_RESPONSE)), dict(text=text)])
    authm = self.requests_mock.post(self.TEST_IDP_URL, content=saml2_fixtures.SAML2_ASSERTION)
    self.requests_mock.post(self.TEST_CONSUMER_URL, status_code=302, headers={'Location': self.TEST_SP_URL})
    resp = requests.get(self.TEST_SP_URL, auth=self.get_plugin())
    self.assertEqual(200, resp.status_code)
    self.assertEqual(text, resp.text)
    self.assertEqual(self.calls, [self.TEST_SP_URL, self.TEST_IDP_URL, self.TEST_CONSUMER_URL, self.TEST_SP_URL])
    self.assertEqual(self.basic_header(), authm.last_request.headers['Authorization'])
    authn_request = self.requests_mock.request_history[1].text
    self.assertThat(saml2_fixtures.AUTHN_REQUEST, matchers.XMLEquals(authn_request))