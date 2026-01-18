import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
def test_trust_scope(self):
    token_id, token = self.get_token(project=False)
    token.set_trust_scope()
    plugin = self.get_plugin(token_id)
    self.assertEqual(token.trust_id, plugin.user.trust_id)
    self.assertEqual(token.trustor_user_id, plugin.user.trustor_user_id)
    self.assertEqual(token.trustee_user_id, plugin.user.trustee_user_id)