import datetime
from unittest import mock
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient import utils as client_utils
def test_set_and_get_keyring_expired(self):
    with self.deprecations.expect_deprecations_here():
        cl = httpclient.HTTPClient(username=USERNAME, password=PASSWORD, project_id=TENANT_ID, auth_url=AUTH_URL, use_keyring=True)
    auth_ref = access.AccessInfo.factory(body=PROJECT_SCOPED_TOKEN)
    expired = timeutils.utcnow() - datetime.timedelta(minutes=30)
    auth_ref['token']['expires'] = client_utils.isotime(expired)
    self.memory_keyring.password = pickle.dumps(auth_ref)
    method = 'get_raw_token_from_identity_service'
    with mock.patch.object(cl, method) as meth:
        meth.return_value = (True, PROJECT_SCOPED_TOKEN)
        self.assertTrue(cl.authenticate())
        self.assertEqual(1, meth.call_count)
    self.assertTrue(self.memory_keyring.fetched)
    new_auth_ref = pickle.loads(self.memory_keyring.password)
    self.assertEqual(new_auth_ref['token']['expires'], PROJECT_SCOPED_TOKEN['access']['token']['expires'])