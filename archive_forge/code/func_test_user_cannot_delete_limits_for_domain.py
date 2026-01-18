import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_delete_limits_for_domain(self):
    _, domain_limit_id = _create_limits_and_dependencies(domain_id=self.domain_id)
    with self.test_client() as c:
        c.delete('/v3/limits/%s' % domain_limit_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)