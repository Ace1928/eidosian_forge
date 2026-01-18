import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def test_send_ecp_authn_response(self):
    self._mock_k2k_flow_urls()
    response = self.k2kplugin._send_service_provider_ecp_authn_response(self.session, self.SP_URL, self.SP_AUTH_URL)
    self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, response.headers['X-Subject-Token'])