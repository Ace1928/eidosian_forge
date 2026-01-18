import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_add_endpoint_group_to_project(self):
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
    endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
    endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
    with self.test_client() as c:
        c.put('/v3/OS-EP-FILTER/endpoint_groups/%s/projects/%s' % (endpoint_group['id'], project['id']), headers=self.headers)