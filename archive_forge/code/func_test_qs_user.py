from openstack.tests.functional import base
def test_qs_user(self):
    sot = self.conn.compute.get_quota_set(self.conn.current_project_id, user_id=self.conn.session.auth.get_user_id(self.conn.compute))
    self.assertIsNotNone(sot.key_pairs)