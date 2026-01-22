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
class MappingTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(MappingTests, self).setUp()
        self.key = 'mapping'
        self.collection_key = 'mappings'
        self.model = mappings.Mapping
        self.manager = self.client.federation.mappings
        self.path_prefix = 'OS-FEDERATION'

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault('rules', [uuid.uuid4().hex, uuid.uuid4().hex])
        return kwargs

    def test_create(self):
        ref = self.new_ref()
        manager_ref = ref.copy()
        mapping_id = manager_ref.pop('id')
        req_ref = ref.copy()
        self.stub_entity('PUT', entity=req_ref, id=mapping_id, status_code=201)
        returned = self.manager.create(mapping_id=mapping_id, **manager_ref)
        self.assertIsInstance(returned, self.model)
        for attr in req_ref:
            self.assertEqual(getattr(returned, attr), req_ref[attr], 'Expected different %s' % attr)
        self.assertEntityRequestBodyIs(manager_ref)