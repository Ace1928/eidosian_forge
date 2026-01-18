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
def test_user_can_create_group_for_own_domain(self):
    create = {'group': {'name': uuid.uuid4().hex, 'domain_id': self.domain_id}}
    with self.test_client() as c:
        c.post('/v3/groups', json=create, headers=self.headers)