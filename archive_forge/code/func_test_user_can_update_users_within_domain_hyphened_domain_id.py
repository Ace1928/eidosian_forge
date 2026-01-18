import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import user as up
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_update_users_within_domain_hyphened_domain_id(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
    update = {'user': {'domain-id': domain['id']}}
    with self.test_client() as c:
        r = c.patch('/v3/users/%s' % user['id'], json=update, headers=self.headers)
        self.assertEqual(domain['id'], r.json['user']['domain-id'])
        self.assertEqual(self.domain_id, r.json['user']['domain_id'])