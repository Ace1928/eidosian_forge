import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.identity import access as access_plugin
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_project_auth_properties(self):
    plugin = self._plugin()
    auth_ref = plugin.auth_ref
    self.assertIsNone(auth_ref.project_domain_id)
    self.assertIsNone(auth_ref.project_domain_name)
    self.assertIsNone(auth_ref.project_id)
    self.assertIsNone(auth_ref.project_name)