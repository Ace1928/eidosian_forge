import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemReaderAndMemberPoliciesAssociationTests(object):

    def test_user_cannot_create_policy_association_for_endpoint(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.put('/v3/policies/%s/OS-ENDPOINT-POLICY/endpoints/%s' % (policy['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_policy_association_for_endpoint(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.delete('/v3/policies/%s/OS-ENDPOINT-POLICY/endpoints/%s' % (policy['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_create_policy_association_for_service(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        with self.test_client() as c:
            c.put('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s' % (policy['id'], service['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_policy_association_for_service(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        with self.test_client() as c:
            c.delete('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s' % (policy['id'], service['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_create_policy_assoc_for_region_and_service(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
        with self.test_client() as c:
            c.put('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s/regions/%s' % (policy['id'], service['id'], region['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_policy_assoc_for_region_and_service(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
        with self.test_client() as c:
            c.delete('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s/regions/%s' % (policy['id'], service['id'], region['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)