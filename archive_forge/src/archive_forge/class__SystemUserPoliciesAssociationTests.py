import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserPoliciesAssociationTests(object):
    """Common default functionality for all system users."""

    def test_user_can_check_policy_association_for_endpoint(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.endpoint_policy_api.create_policy_association(policy['id'], endpoint['id'])
        with self.test_client() as c:
            c.get('/v3/policies/%s/OS-ENDPOINT-POLICY/endpoints/%s' % (policy['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_can_check_policy_association_for_service(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        PROVIDERS.endpoint_policy_api.create_policy_association(policy['id'], service_id=service['id'])
        with self.test_client() as c:
            c.get('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s' % (policy['id'], service['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_can_check_policy_association_for_region_and_service(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
        PROVIDERS.endpoint_policy_api.create_policy_association(policy['id'], service_id=service['id'], region_id=region['id'])
        with self.test_client() as c:
            c.get('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s/regions/%s' % (policy['id'], service['id'], region['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_can_get_policy_for_endpoint(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.endpoint_policy_api.create_policy_association(policy['id'], endpoint['id'])
        with self.test_client() as c:
            c.get('/v3/endpoints/%s/OS-ENDPOINT-POLICY/policy' % endpoint['id'], headers=self.headers)

    def test_user_list_endpoints_for_policy(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        PROVIDERS.endpoint_policy_api.create_policy_association(policy['id'], endpoint['id'])
        with self.test_client() as c:
            r = c.get('/v3/policies/%s/OS-ENDPOINT-POLICY/endpoints' % policy['id'], headers=self.headers)
            for endpoint_itr in r.json['endpoints']:
                self.assertIn(endpoint['id'], endpoint_itr['id'])