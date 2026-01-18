import copy
import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
def test_shadow_existing_federated_user(self):
    federated_user1 = copy.deepcopy(self.federated_user)
    ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user1, email=self.email)
    shadow_user1 = PROVIDERS.identity_api.shadow_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], federated_user1)
    self.assertEqual(federated_user1['display_name'], shadow_user1['name'])
    federated_user2 = copy.deepcopy(self.federated_user)
    federated_user2['display_name'] = uuid.uuid4().hex
    ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user2, email=self.email)
    shadow_user2 = PROVIDERS.identity_api.shadow_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], federated_user2)
    self.assertEqual(federated_user2['display_name'], shadow_user2['name'])
    self.assertNotEqual(shadow_user1['name'], shadow_user2['name'])
    self.assertEqual(shadow_user1['id'], shadow_user2['id'])