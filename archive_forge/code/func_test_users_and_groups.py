from openstack import exceptions
from openstack.tests.functional import base
def test_users_and_groups(self):
    i_ver = self.operator_cloud.config.get_api_version('identity')
    if i_ver in ('2', '2.0'):
        self.skipTest('Identity service does not support groups')
    group_name = self.getUniqueString('group')
    self.addCleanup(self.operator_cloud.delete_group, group_name)
    group = self.operator_cloud.create_group(group_name, 'test group')
    self.assertIsNotNone(group)
    user_name = self.user_prefix + '_ug'
    user_email = 'nobody@nowhere.com'
    user = self._create_user(name=user_name, email=user_email)
    self.assertIsNotNone(user)
    self.operator_cloud.add_user_to_group(user_name, group_name)
    self.assertTrue(self.operator_cloud.is_user_in_group(user_name, group_name))
    self.operator_cloud.remove_user_from_group(user_name, group_name)
    self.assertFalse(self.operator_cloud.is_user_in_group(user_name, group_name))