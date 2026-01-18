import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_service_providers_urls(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)
    s = session.Session()
    auth_ref = a.get_auth_ref(s)
    service_providers = auth_ref.service_providers
    self.assertEqual('https://sp1.com/v3/OS-FEDERATION/identity_providers/acme/protocols/saml2/auth', service_providers.get_auth_url('sp1'))
    self.assertEqual('https://sp1.com/Shibboleth.sso/SAML2/ECP', service_providers.get_sp_url('sp1'))
    self.assertEqual('https://sp2.com/v3/OS-FEDERATION/identity_providers/acme/protocols/saml2/auth', service_providers.get_auth_url('sp2'))
    self.assertEqual('https://sp2.com/Shibboleth.sso/SAML2/ECP', service_providers.get_sp_url('sp2'))