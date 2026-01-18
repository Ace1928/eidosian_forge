import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import fixture
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_scoped_init(self):
    token = client_fixtures.project_scoped_token()
    self.stub_auth(json=token)
    with self.deprecations.expect_deprecations_here():
        c = client.Client(username='exampleuser', password='password', project_name='exampleproject', auth_url=self.TEST_URL)
    self.assertIsNotNone(c.auth_ref)
    with self.deprecations.expect_deprecations_here():
        self.assertTrue(c.auth_ref.scoped)
    self.assertTrue(c.auth_ref.project_scoped)
    self.assertFalse(c.auth_ref.domain_scoped)
    self.assertIsNone(c.auth_ref.trust_id)
    self.assertFalse(c.auth_ref.trust_scoped)
    self.assertEqual(token.tenant_id, c.get_project_id(session=None))
    self.assertEqual(token.user_id, c.get_user_id(session=None))