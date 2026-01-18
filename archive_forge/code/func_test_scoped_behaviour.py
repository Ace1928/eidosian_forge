import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def test_scoped_behaviour(self):
    auth = self.get_plugin(project_id=self.scoped_token.project_id)
    sess = session.Session(auth=auth)
    self.assertEqual(self.scoped_token_id, sess.get_token())
    self.assertTrue(self.unscoped_mock.called)
    self.assertTrue(self.scoped_mock.called)