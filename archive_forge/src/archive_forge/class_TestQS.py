from openstack.tests.functional import base
class TestQS(base.BaseFunctionalTest):

    def test_qs(self):
        sot = self.conn.compute.get_quota_set(self.conn.current_project_id)
        self.assertIsNotNone(sot.key_pairs)

    def test_qs_user(self):
        sot = self.conn.compute.get_quota_set(self.conn.current_project_id, user_id=self.conn.session.auth.get_user_id(self.conn.compute))
        self.assertIsNotNone(sot.key_pairs)

    def test_update(self):
        sot = self.conn.compute.get_quota_set(self.conn.current_project_id)
        self.conn.compute.update_quota_set(sot, query={'user_id': self.conn.session.auth.get_user_id(self.conn.compute)}, key_pairs=100)

    def test_revert(self):
        self.conn.compute.revert_quota_set(self.conn.current_project_id, user_id=self.conn.session.auth.get_user_id(self.conn.compute))