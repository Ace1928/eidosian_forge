import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_check_endpoint_in_project(self):
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
    endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    PROVIDERS.catalog_api.add_endpoint_to_project(endpoint['id'], project['id'])
    with self.test_client() as c:
        c.get('/v3/OS-EP-FILTER/projects/%s/endpoints/%s' % (project['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)