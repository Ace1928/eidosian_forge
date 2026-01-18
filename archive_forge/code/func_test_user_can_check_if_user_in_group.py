import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import group as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_check_if_user_in_group(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
    PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
    with self.test_client() as c:
        c.get('/v3/groups/%s/users/%s' % (group['id'], user['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)