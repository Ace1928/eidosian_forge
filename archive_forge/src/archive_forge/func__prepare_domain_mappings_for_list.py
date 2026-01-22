import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def _prepare_domain_mappings_for_list(self):
    local_entities = [{'domain_id': self.domainA['id'], 'entity_type': mapping.EntityType.USER}, {'domain_id': self.domainA['id'], 'entity_type': mapping.EntityType.USER}, {'domain_id': self.domainB['id'], 'entity_type': mapping.EntityType.GROUP}, {'domain_id': self.domainB['id'], 'entity_type': mapping.EntityType.USER}, {'domain_id': self.domainB['id'], 'entity_type': mapping.EntityType.USER}]
    for e in local_entities:
        e['local_id'] = uuid.uuid4().hex
        e['public_id'] = PROVIDERS.id_mapping_api.create_id_mapping(e)
    return local_entities