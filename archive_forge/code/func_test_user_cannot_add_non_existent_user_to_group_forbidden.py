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
def test_user_cannot_add_non_existent_user_to_group_forbidden(self):
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
    with self.test_client() as c:
        c.put('/v3/groups/%(group)s/users/%(user)s' % {'group': group['id'], 'user': uuid.uuid4().hex}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)