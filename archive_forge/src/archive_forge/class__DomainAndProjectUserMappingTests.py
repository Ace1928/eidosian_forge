import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _DomainAndProjectUserMappingTests(object):

    def test_user_cannot_create_mappings(self):
        create = {'mapping': {'id': uuid.uuid4().hex, 'rules': [{'local': [{'user': {'name': '{0}'}}], 'remote': [{'type': 'UserName'}]}]}}
        mapping_id = create['mapping']['id']
        with self.test_client() as c:
            c.put('/v3/OS-FEDERATION/mappings/%s' % mapping_id, json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_mappings(self):
        mapping = unit.new_mapping_ref()
        mapping = PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
        with self.test_client() as c:
            c.get('/v3/OS-FEDERATION/mappings', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_a_mapping(self):
        mapping = unit.new_mapping_ref()
        mapping = PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
        with self.test_client() as c:
            c.get('/v3/OS-FEDERATION/mappings/%s' % mapping['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_mappings(self):
        mapping = unit.new_mapping_ref()
        mapping = PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
        update = {'mapping': {'rules': [{'local': [{'user': {'name': '{0}'}}], 'remote': [{'type': 'UserName'}]}]}}
        with self.test_client() as c:
            c.patch('/v3/OS-FEDERATION/mappings/%s' % mapping['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_mappings(self):
        mapping = unit.new_mapping_ref()
        mapping = PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
        with self.test_client() as c:
            c.delete('/v3/OS-FEDERATION/mappings/%s' % mapping['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)