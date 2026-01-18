from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_token_no_id_ex(self):
    p = {'identity': {'methods': ['token'], 'token': {}}}
    self._expect_failure(p)