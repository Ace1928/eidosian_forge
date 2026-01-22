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
def _test_delete_application_credential(self, expected_status_code=http.client.NO_CONTENT):
    app_cred = self._create_application_credential()
    with self.test_client() as c:
        c.delete('/v3/users/%s/application_credentials/%s' % (self.app_cred_user_id, app_cred['id']), expected_status_code=expected_status_code, headers=self.headers)