import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_list_endpoint_groups(self):
    endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
    PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
    with self.test_client() as c:
        c.get('/v3/OS-EP-FILTER/endpoint_groups', headers=self.headers, expected_status_code=http.client.FORBIDDEN)