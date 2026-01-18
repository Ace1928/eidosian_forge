import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_domain_limits(self):
    _, domain_limit_id = _create_limits_and_dependencies(domain_id=self.domain_id)
    with self.test_client() as c:
        r = c.get('/v3/limits/%s' % domain_limit_id, headers=self.headers)
        self.assertEqual(self.domain_id, r.json['limit']['domain_id'])