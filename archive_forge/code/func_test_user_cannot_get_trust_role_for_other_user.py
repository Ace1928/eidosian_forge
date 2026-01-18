import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_get_trust_role_for_other_user(self):
    PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
    with self.test_client() as c:
        c.head('/v3/OS-TRUST/trusts/%s/roles/%s' % (self.trust_id, self.bootstrapper.member_role_id), headers=self.other_headers, expected_status_code=http.client.FORBIDDEN)