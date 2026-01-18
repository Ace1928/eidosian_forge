import datetime
import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as base_policy
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_list_application_credentials(self):
    self._create_application_credential()
    self._create_application_credential()
    with self.test_client() as c:
        c.get('/v3/users/%s/application_credentials' % self.app_cred_user_id, expected_status_code=http.client.FORBIDDEN, headers=self.headers)