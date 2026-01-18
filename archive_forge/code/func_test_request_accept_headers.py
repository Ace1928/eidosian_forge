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
def test_request_accept_headers(self):
    random_header = uuid.uuid4().hex
    headers = {'Accept': random_header}
    req = requests.Request('GET', 'http://another.test', headers=headers)
    plugin = self.get_plugin()
    plugin_headers = plugin(req).headers
    self.assertIn('Accept', plugin_headers)
    accept_header = plugin_headers['Accept']
    self.assertIn(self.HEADER_MEDIA_TYPE_SEPARATOR, accept_header)
    self.assertIn(random_header, accept_header.split(self.HEADER_MEDIA_TYPE_SEPARATOR))
    self.assertIn(PAOS_HEADER, accept_header.split(self.HEADER_MEDIA_TYPE_SEPARATOR))