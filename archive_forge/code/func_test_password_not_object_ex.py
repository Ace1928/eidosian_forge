from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_password_not_object_ex(self):
    p = {'identity': {'methods': ['password'], 'password': 'something'}}
    self._expect_failure(p)