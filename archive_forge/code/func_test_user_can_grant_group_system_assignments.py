import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_grant_group_system_assignments(self):
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(CONF.identity.default_domain_id))
    with self.test_client() as c:
        c.put('/v3/system/groups/%s/roles/%s' % (group['id'], self.bootstrapper.member_role_id), headers=self.headers)