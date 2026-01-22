import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def _test_v3_headers(self, token, prefix):
    self.assertEqual(token.domain_id, self.request.headers['X%s-Domain-Id' % prefix])
    self.assertEqual(token.domain_name, self.request.headers['X%s-Domain-Name' % prefix])
    self.assertEqual(token.project_id, self.request.headers['X%s-Project-Id' % prefix])
    self.assertEqual(token.project_name, self.request.headers['X%s-Project-Name' % prefix])
    self.assertEqual(token.project_domain_id, self.request.headers['X%s-Project-Domain-Id' % prefix])
    self.assertEqual(token.project_domain_name, self.request.headers['X%s-Project-Domain-Name' % prefix])
    self.assertEqual(token.user_id, self.request.headers['X%s-User-Id' % prefix])
    self.assertEqual(token.user_name, self.request.headers['X%s-User-Name' % prefix])
    self.assertEqual(token.user_domain_id, self.request.headers['X%s-User-Domain-Id' % prefix])
    self.assertEqual(token.user_domain_name, self.request.headers['X%s-User-Domain-Name' % prefix])