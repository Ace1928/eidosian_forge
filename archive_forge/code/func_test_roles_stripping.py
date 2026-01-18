import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_roles_stripping(self):
    req = self._build_request(roles=['\trole1'])
    self._build_middleware().process_request(req)
    self.assertIn('role1', req.context.roles)