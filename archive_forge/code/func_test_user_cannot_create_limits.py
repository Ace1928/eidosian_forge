import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_create_limits(self):
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
    registered_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
    registered_limit = registered_limits[0]
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
    create = {'limits': [unit.new_limit_ref(project_id=project['id'], service_id=service['id'], resource_name=registered_limit['resource_name'], resource_limit=5)]}
    with self.test_client() as c:
        c.post('/v3/limits', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)