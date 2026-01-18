import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import project as pp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_update_non_existent_project_forbidden(self):
    update = {'project': {'description': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.patch('/v3/projects/%s' % uuid.uuid4().hex, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)