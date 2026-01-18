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
def test_user_cannot_list_groups_for_non_existent_user_forbidden(self):
    with self.test_client() as c:
        c.get('/v3/users/%s/groups' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)