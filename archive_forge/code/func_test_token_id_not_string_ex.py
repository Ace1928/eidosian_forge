from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_token_id_not_string_ex(self):
    p = {'identity': {'methods': ['token'], 'token': {'id': 123}}}
    self._expect_failure(p)