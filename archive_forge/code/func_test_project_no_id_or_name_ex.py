from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_project_no_id_or_name_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'project': {}}}
    self._expect_failure(p)