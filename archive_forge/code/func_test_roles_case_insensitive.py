import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_roles_case_insensitive(self):
    req = self._build_request(roles=['Admin', 'role2'])
    self._build_middleware().process_request(req)
    self.assertTrue(req.context.is_admin)