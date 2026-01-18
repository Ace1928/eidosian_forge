from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_project_not_object_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'project': 'something'}}
    self._expect_failure(p)