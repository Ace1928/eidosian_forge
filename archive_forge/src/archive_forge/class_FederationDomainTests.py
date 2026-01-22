import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
class FederationDomainTests(utils.ClientTestCase):

    def setUp(self):
        super(FederationDomainTests, self).setUp()
        self.key = 'domain'
        self.collection_key = 'domains'
        self.model = domains.Domain
        self.manager = self.client.federation.domains
        self.URL = '%s%s' % (self.TEST_URL, '/auth/domains')

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault('enabled', True)
        kwargs.setdefault('name', uuid.uuid4().hex)
        kwargs.setdefault('description', uuid.uuid4().hex)
        return kwargs

    def test_list_accessible_domains(self):
        domains_ref = [self.new_ref(), self.new_ref()]
        domains_json = {self.collection_key: domains_ref}
        self.requests_mock.get(self.URL, json=domains_json)
        returned_list = self.manager.list()
        self.assertEqual(len(domains_ref), len(returned_list))
        for domain in returned_list:
            self.assertIsInstance(domain, self.model)