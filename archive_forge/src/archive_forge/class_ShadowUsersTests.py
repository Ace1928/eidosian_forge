import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity.shadow_users import test_backend
from keystone.tests.unit.identity.shadow_users import test_core
from keystone.tests.unit.ksfixtures import database
class ShadowUsersTests(unit.TestCase, test_backend.ShadowUsersBackendTests, test_core.ShadowUsersCoreTests):

    def setUp(self):
        super(ShadowUsersTests, self).setUp()
        self.useFixture(database.Database())
        self.load_backends()
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
        self.idp = {'id': uuid.uuid4().hex, 'enabled': True, 'description': uuid.uuid4().hex}
        self.mapping = {'id': uuid.uuid4().hex}
        self.protocol = {'id': uuid.uuid4().hex, 'idp_id': self.idp['id'], 'mapping_id': self.mapping['id']}
        self.federated_user = {'idp_id': self.idp['id'], 'protocol_id': self.protocol['id'], 'unique_id': uuid.uuid4().hex, 'display_name': uuid.uuid4().hex}
        self.email = uuid.uuid4().hex
        PROVIDERS.federation_api.create_idp(self.idp['id'], self.idp)
        PROVIDERS.federation_api.create_mapping(self.mapping['id'], self.mapping)
        PROVIDERS.federation_api.create_protocol(self.idp['id'], self.protocol['id'], self.protocol)
        self.domain_id = PROVIDERS.federation_api.get_idp(self.idp['id'])['domain_id']