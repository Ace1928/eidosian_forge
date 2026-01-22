import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
class SqlIDMapping(test_backend_sql.SqlTests):

    def setUp(self):
        super(SqlIDMapping, self).setUp()
        self.load_sample_data()

    def load_sample_data(self):
        self.addCleanup(self.clean_sample_data)
        domainA = unit.new_domain_ref()
        self.domainA = PROVIDERS.resource_api.create_domain(domainA['id'], domainA)
        domainB = unit.new_domain_ref()
        self.domainB = PROVIDERS.resource_api.create_domain(domainB['id'], domainB)

    def clean_sample_data(self):
        if hasattr(self, 'domainA'):
            self.domainA['enabled'] = False
            PROVIDERS.resource_api.update_domain(self.domainA['id'], self.domainA)
            PROVIDERS.resource_api.delete_domain(self.domainA['id'])
        if hasattr(self, 'domainB'):
            self.domainB['enabled'] = False
            PROVIDERS.resource_api.update_domain(self.domainB['id'], self.domainB)
            PROVIDERS.resource_api.delete_domain(self.domainB['id'])

    def test_invalid_public_key(self):
        self.assertIsNone(PROVIDERS.id_mapping_api.get_id_mapping(uuid.uuid4().hex))

    def test_id_mapping_crud(self):
        initial_mappings = len(mapping_sql.list_id_mappings())
        local_id1 = uuid.uuid4().hex
        local_id2 = uuid.uuid4().hex
        local_entity1 = {'domain_id': self.domainA['id'], 'local_id': local_id1, 'entity_type': mapping.EntityType.USER}
        local_entity2 = {'domain_id': self.domainB['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.GROUP}
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity1))
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity2))
        public_id1 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity1)
        public_id2 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity2)
        self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 2))
        self.assertEqual(public_id1, PROVIDERS.id_mapping_api.get_public_id(local_entity1))
        self.assertEqual(public_id2, PROVIDERS.id_mapping_api.get_public_id(local_entity2))
        local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id1)
        self.assertEqual(self.domainA['id'], local_id_ref['domain_id'])
        self.assertEqual(local_id1, local_id_ref['local_id'])
        self.assertEqual(mapping.EntityType.USER, local_id_ref['entity_type'])
        self.assertNotEqual(local_id1, public_id1)
        local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id2)
        self.assertEqual(self.domainB['id'], local_id_ref['domain_id'])
        self.assertEqual(local_id2, local_id_ref['local_id'])
        self.assertEqual(mapping.EntityType.GROUP, local_id_ref['entity_type'])
        self.assertNotEqual(local_id2, public_id2)
        new_public_id = uuid.uuid4().hex
        public_id3 = PROVIDERS.id_mapping_api.create_id_mapping({'domain_id': self.domainB['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.USER}, public_id=new_public_id)
        self.assertEqual(new_public_id, public_id3)
        self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 3))
        PROVIDERS.id_mapping_api.delete_id_mapping(public_id1)
        PROVIDERS.id_mapping_api.delete_id_mapping(public_id2)
        PROVIDERS.id_mapping_api.delete_id_mapping(public_id3)
        self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings))

    def test_id_mapping_handles_unicode(self):
        initial_mappings = len(mapping_sql.list_id_mappings())
        local_id = u'fäké1'
        local_entity = {'domain_id': self.domainA['id'], 'local_id': local_id, 'entity_type': mapping.EntityType.USER}
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
        public_id = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
        self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 1))
        self.assertEqual(public_id, PROVIDERS.id_mapping_api.get_public_id(local_entity))

    def test_id_mapping_handles_bytes(self):
        initial_mappings = len(mapping_sql.list_id_mappings())
        local_id = b'FaKeID'
        local_entity = {'domain_id': self.domainA['id'], 'local_id': local_id, 'entity_type': mapping.EntityType.USER}
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
        public_id = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
        self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 1))
        self.assertEqual(public_id, PROVIDERS.id_mapping_api.get_public_id(local_entity))

    def test_id_mapping_handles_ids_greater_than_64_characters(self):
        initial_mappings = len(mapping_sql.list_id_mappings())
        local_id = 'Aa' * 100
        local_entity = {'domain_id': self.domainA['id'], 'local_id': local_id, 'entity_type': mapping.EntityType.GROUP}
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
        public_id = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
        self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 1))
        self.assertEqual(public_id, PROVIDERS.id_mapping_api.get_public_id(local_entity))
        self.assertEqual(local_id, PROVIDERS.id_mapping_api.get_id_mapping(public_id)['local_id'])

    def test_delete_public_id_is_silent(self):
        PROVIDERS.id_mapping_api.delete_id_mapping(uuid.uuid4().hex)

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

    def test_create_duplicate_mapping(self):
        local_entity = {'domain_id': self.domainA['id'], 'local_id': uuid.uuid4().hex, 'entity_type': mapping.EntityType.USER}
        public_id1 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
        public_id2 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
        self.assertEqual(public_id1, public_id2)
        public_id3 = PROVIDERS.id_mapping_api.create_id_mapping(local_entity, public_id=uuid.uuid4().hex)
        self.assertEqual(public_id1, public_id3)

    @unit.skip_if_cache_disabled('identity')
    def test_cache_when_id_mapping_crud(self):
        local_id = uuid.uuid4().hex
        local_entity = {'domain_id': self.domainA['id'], 'local_id': local_id, 'entity_type': mapping.EntityType.USER}
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
        public_id = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
        self.assertEqual(public_id, PROVIDERS.id_mapping_api.get_public_id(local_entity))
        local_id_ref = PROVIDERS.id_mapping_api.get_id_mapping(public_id)
        self.assertEqual(self.domainA['id'], local_id_ref['domain_id'])
        self.assertEqual(local_id, local_id_ref['local_id'])
        self.assertEqual(mapping.EntityType.USER, local_id_ref['entity_type'])
        PROVIDERS.id_mapping_api.delete_id_mapping(public_id)
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
        self.assertIsNone(PROVIDERS.id_mapping_api.get_id_mapping(public_id))

    @unit.skip_if_cache_disabled('identity')
    def test_invalidate_cache_when_purge_mappings(self):
        local_id1 = uuid.uuid4().hex
        local_id2 = uuid.uuid4().hex
        local_id3 = uuid.uuid4().hex
        local_id4 = uuid.uuid4().hex
        local_id5 = uuid.uuid4().hex
        local_entity1 = {'domain_id': self.domainA['id'], 'local_id': local_id1, 'entity_type': mapping.EntityType.USER}
        local_entity2 = {'domain_id': self.domainA['id'], 'local_id': local_id2, 'entity_type': mapping.EntityType.USER}
        local_entity3 = {'domain_id': self.domainB['id'], 'local_id': local_id3, 'entity_type': mapping.EntityType.GROUP}
        local_entity4 = {'domain_id': self.domainB['id'], 'local_id': local_id4, 'entity_type': mapping.EntityType.USER}
        local_entity5 = {'domain_id': self.domainB['id'], 'local_id': local_id5, 'entity_type': mapping.EntityType.USER}
        PROVIDERS.id_mapping_api.create_id_mapping(local_entity1)
        PROVIDERS.id_mapping_api.create_id_mapping(local_entity2)
        PROVIDERS.id_mapping_api.create_id_mapping(local_entity3)
        PROVIDERS.id_mapping_api.create_id_mapping(local_entity4)
        PROVIDERS.id_mapping_api.create_id_mapping(local_entity5)
        PROVIDERS.id_mapping_api.purge_mappings({'domain_id': self.domainA['id']})
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity1))
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity2))
        PROVIDERS.id_mapping_api.purge_mappings({'entity_type': mapping.EntityType.GROUP})
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity3))
        PROVIDERS.id_mapping_api.purge_mappings({'domain_id': self.domainB['id'], 'local_id': local_id4, 'entity_type': mapping.EntityType.USER})
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity4))
        PROVIDERS.id_mapping_api.purge_mappings({})
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity5))

    def _prepare_domain_mappings_for_list(self):
        local_entities = [{'domain_id': self.domainA['id'], 'entity_type': mapping.EntityType.USER}, {'domain_id': self.domainA['id'], 'entity_type': mapping.EntityType.USER}, {'domain_id': self.domainB['id'], 'entity_type': mapping.EntityType.GROUP}, {'domain_id': self.domainB['id'], 'entity_type': mapping.EntityType.USER}, {'domain_id': self.domainB['id'], 'entity_type': mapping.EntityType.USER}]
        for e in local_entities:
            e['local_id'] = uuid.uuid4().hex
            e['public_id'] = PROVIDERS.id_mapping_api.create_id_mapping(e)
        return local_entities

    def test_get_domain_mapping_list(self):
        local_entities = self._prepare_domain_mappings_for_list()
        with sql.session_for_read():
            domain_a_mappings = PROVIDERS.id_mapping_api.get_domain_mapping_list(self.domainA['id'])
            domain_a_mappings = [m.to_dict() for m in domain_a_mappings]
        self.assertCountEqual(local_entities[:2], domain_a_mappings)

    def test_get_domain_mapping_list_by_user_entity_type(self):
        local_entities = self._prepare_domain_mappings_for_list()
        with sql.session_for_read():
            domain_b_mappings_user = PROVIDERS.id_mapping_api.get_domain_mapping_list(self.domainB['id'], entity_type=mapping.EntityType.USER)
            domain_b_mappings_user = [m.to_dict() for m in domain_b_mappings_user]
        self.assertCountEqual(local_entities[-2:], domain_b_mappings_user)

    def test_get_domain_mapping_list_by_group_entity_type(self):
        local_entities = self._prepare_domain_mappings_for_list()
        with sql.session_for_read():
            domain_b_mappings_group = PROVIDERS.id_mapping_api.get_domain_mapping_list(self.domainB['id'], entity_type=mapping.EntityType.GROUP)
            domain_b_mappings_group = domain_b_mappings_group.first().to_dict()
        self.assertCountEqual(local_entities[2], domain_b_mappings_group)