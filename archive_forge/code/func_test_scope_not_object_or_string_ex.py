from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_scope_not_object_or_string_ex(self):
    p = {'identity': {'methods': []}, 'scope': 1}
    self._expect_failure(p)