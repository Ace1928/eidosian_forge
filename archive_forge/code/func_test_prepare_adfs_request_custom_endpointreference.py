import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_prepare_adfs_request_custom_endpointreference(self):
    self.adfsplugin = saml2.V3ADFSPassword(self.TEST_URL, self.IDENTITY_PROVIDER, self.IDENTITY_PROVIDER_URL, self.SP_ENDPOINT, self.TEST_USER, self.TEST_TOKEN, self.PROTOCOL, self.SP_ENTITYID)
    self.adfsplugin._prepare_adfs_request()
    address = self.adfsplugin.prepared_request.xpath(self.ADDRESS_XPATH, namespaces=self.NAMESPACES)[0]
    self.assertEqual(self.SP_ENTITYID, address.text)