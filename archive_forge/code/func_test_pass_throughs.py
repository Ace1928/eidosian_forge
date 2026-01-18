import uuid
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import service_token
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_pass_throughs(self):
    self.assertEqual(self.user_auth.get_token(self.session), self.combined_auth.get_token(self.session))
    self.assertEqual(self.user_auth.get_endpoint(self.session, 'identity'), self.combined_auth.get_endpoint(self.session, 'identity'))
    self.assertEqual(self.user_auth.get_user_id(self.session), self.combined_auth.get_user_id(self.session))
    self.assertEqual(self.user_auth.get_project_id(self.session), self.combined_auth.get_project_id(self.session))
    self.assertEqual(self.user_auth.get_sp_auth_url(self.session, 'a'), self.combined_auth.get_sp_auth_url(self.session, 'a'))
    self.assertEqual(self.user_auth.get_sp_url(self.session, 'a'), self.combined_auth.get_sp_url(self.session, 'a'))