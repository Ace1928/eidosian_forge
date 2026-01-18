from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_password_user_not_object_ex(self):
    p = {'identity': {'methods': ['password'], 'password': {'user': 'something'}}}
    self._expect_failure(p)