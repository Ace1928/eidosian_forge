import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemReaderAndMemberImpliedRoleTests(object):
    """Common default functionality for system readers and system members."""

    def test_user_cannot_create_implied_roles(self):
        with self.test_client() as c:
            c.put('/v3/roles/%s/implies/%s' % (self.prior_role_id, self.implied_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_implied_roles(self):
        PROVIDERS.role_api.create_implied_role(self.prior_role_id, self.implied_role_id)
        with self.test_client() as c:
            c.delete('/v3/roles/%s/implies/%s' % (self.prior_role_id, self.implied_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)