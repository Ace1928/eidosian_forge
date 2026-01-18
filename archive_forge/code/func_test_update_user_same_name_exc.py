import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_user_same_name_exc(self):
    if not self.allows_name_update:
        self.skipTest("Backend doesn't allow name update.")
    domain_id = uuid.uuid4().hex
    user1 = self.create_user(domain_id=domain_id)
    user2 = self.create_user(domain_id=domain_id)
    user_mod = {'name': user2['name']}
    self.assertRaises(exception.Conflict, self.driver.update_user, user1['id'], user_mod)