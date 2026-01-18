import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_prepare_sp_request(self):
    assertion = etree.XML(self.ADFS_SECURITY_TOKEN_RESPONSE)
    assertion = assertion.xpath(saml2.V3ADFSPassword.ADFS_ASSERTION_XPATH, namespaces=saml2.V3ADFSPassword.ADFS_TOKEN_NAMESPACES)
    assertion = assertion[0]
    assertion = etree.tostring(assertion)
    assertion = assertion.replace(b'http://docs.oasis-open.org/ws-sx/ws-trust/200512', b'http://schemas.xmlsoap.org/ws/2005/02/trust')
    assertion = urllib.parse.quote(assertion)
    assertion = 'wa=wsignin1.0&wresult=' + assertion
    self.adfsplugin.adfs_token = etree.XML(self.ADFS_SECURITY_TOKEN_RESPONSE)
    self.adfsplugin._prepare_sp_request()
    self.assertEqual(assertion, self.adfsplugin.encoded_assertion)