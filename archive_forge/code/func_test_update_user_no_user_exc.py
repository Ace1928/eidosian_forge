import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_user_no_user_exc(self):
    user_id = uuid.uuid4().hex
    user_mod = {'enabled': False}
    self.assertRaises(exception.UserNotFound, self.driver.update_user, user_id, user_mod)