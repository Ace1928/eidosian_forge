import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_check_policy_association_for_region_and_service(self):
    policy = unit.new_policy_ref()
    policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
    PROVIDERS.endpoint_policy_api.create_policy_association(policy['id'], service_id=service['id'], region_id=region['id'])
    with self.test_client() as c:
        c.get('/v3/policies/%s/OS-ENDPOINT-POLICY/services/%s/regions/%s' % (policy['id'], service['id'], region['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)