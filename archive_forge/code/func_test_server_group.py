from openstack.tests.functional import base
def test_server_group(self):
    server_group_name = self.getUniqueString()
    self.addCleanup(self.cleanup, server_group_name)
    server_group = self.user_cloud.create_server_group(server_group_name, ['affinity'])
    server_group_ids = [v['id'] for v in self.user_cloud.list_server_groups()]
    self.assertIn(server_group['id'], server_group_ids)
    self.user_cloud.delete_server_group(server_group_name)