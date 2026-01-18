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
def test_passed_when_not_200(self):
    text = uuid.uuid4().hex
    test_url = 'http://another.test'
    self.requests_mock.get(test_url, status_code=201, headers=CONTENT_TYPE_PAOS_HEADER, text=text)
    resp = requests.get(test_url, auth=self.get_plugin())
    self.assertEqual(201, resp.status_code)
    self.assertEqual(text, resp.text)