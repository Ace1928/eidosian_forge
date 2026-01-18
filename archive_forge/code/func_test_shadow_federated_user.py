import copy
import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
def test_shadow_federated_user(self):
    federated_user1 = copy.deepcopy(self.federated_user)
    ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user1, email=self.email)
    user = PROVIDERS.identity_api.shadow_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], federated_user1)
    self.assertIsNotNone(user['id'])
    self.assertEqual(7, len(user.keys()))
    self.assertIsNotNone(user['name'])
    self.assertIsNone(user['password_expires_at'])
    self.assertIsNotNone(user['domain_id'])
    self.assertEqual(True, user['enabled'])
    self.assertIsNotNone(user['email'])