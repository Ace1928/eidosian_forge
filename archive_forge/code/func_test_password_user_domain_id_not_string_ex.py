from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_password_user_domain_id_not_string_ex(self):
    p = {'identity': {'methods': ['password'], 'password': {'user': {'id': 'something', 'domain': {'id': {}}}}}}
    self._expect_failure(p)