import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def test_purge_mappings(self):
    initial_mappings = len(mapping_sql.list_id_mappings())
    local_id1 = uuid.uuid4().hex
    local_id2 = uuid.uuid4().hex
    local_id3 = uuid.uuid4().hex
    local_id4 = uuid.uuid4().hex
    local_id5 = uuid.uuid4().hex
    PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainA['id'], 'local_id': local_id1, 'entity_type': mapping.EntityType.USER})
    PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainA['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.USER})
    public_id3 = PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainB['id'], 'local_id': local_id3, 'entity_type': mapping.EntityType.GROUP})
    public_id4 = PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainB['id'], 'local_id': local_id4, 'entity_type': mapping.EntityType.USER})
    public_id5 = PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainB['id'], 'local_id': local_id5, 'entity_type': mapping.EntityType.USER})
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 5))
    PROVIDERS.id_mapping_api.purge_mappings({'domain_id': self.domainA['id']})
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 3))
    PROVIDERS.id_mapping_api.get_id_mapping(public_id3)
    PROVIDERS.id_mapping_api.get_id_mapping(public_id4)
    PROVIDERS.id_mapping_api.get_id_mapping(public_id5)
    PROVIDERS.id_mapping_api.purge_mappings({'entity_type': mapping.EntityType.GROUP})
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 2))
    PROVIDERS.id_mapping_api.get_id_mapping(public_id4)
    PROVIDERS.id_mapping_api.get_id_mapping(public_id5)
    PROVIDERS.id_mapping_api.purge_mappings({'domain_id': self.domainB['id'], 'local_id': local_id4, 'entity_type': mapping.EntityType.USER})
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 1))
    PROVIDERS.id_mapping_api.get_id_mapping(public_id5)
    PROVIDERS.id_mapping_api.purge_mappings({})
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings))