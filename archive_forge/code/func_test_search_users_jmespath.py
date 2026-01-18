from openstack import exceptions
from openstack.tests.functional import base
def test_search_users_jmespath(self):
    users = self.operator_cloud.search_users(filters='[?enabled]')
    self.assertIsNotNone(users)