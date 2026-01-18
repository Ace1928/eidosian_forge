from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_trust_not_object_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'OS-TRUST:trust': 'something'}}
    self._expect_failure(p)