import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_create_implied_roles(self):
    with self.test_client() as c:
        c.put('/v3/roles/%s/implies/%s' % (self.prior_role_id, self.implied_role_id), headers=self.headers, expected_status_code=http.client.CREATED)