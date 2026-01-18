import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_user_name_not_allowed_exc(self):
    if self.allows_name_update:
        self.skipTest('Backend allows name update.')
    user = self.create_user()
    user_mod = {'name': uuid.uuid4().hex}
    self.assertRaises(exception.Conflict, self.driver.update_user, user['id'], user_mod)