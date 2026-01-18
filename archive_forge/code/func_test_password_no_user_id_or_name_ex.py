from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_password_no_user_id_or_name_ex(self):
    p = {'identity': {'methods': ['password'], 'password': {'user': {}}}}
    self._expect_failure(p)