import uuid
from keystone import exception
def test_update_policy(self):
    self.policy['blob'] = '{"identity:create_user": "role:domain_admin","identity:update_user": "role:domain_admin"}'
    self.driver.update_policy(self.policy['id'], self.policy)
    self.assertEqual(self.policy, self.driver.get_policy(self.policy['id']))