from openstack.tests.functional.shared_file_system import base
class UserMessageTest(base.BaseSharedFileSystemTest):

    def test_user_messages(self):
        u_messages = self.user_cloud.shared_file_system.user_messages()
        for u_message in u_messages:
            for attribute in ('id', 'created_at', 'action_id', 'detail_id', 'expires_at', 'message_level', 'project_id', 'request_id', 'resource_id', 'resource_type', 'user_message'):
                self.assertTrue(hasattr(u_message, attribute))
                self.assertIsInstance(getattr(u_message, attribute), str)
            self.conn.shared_file_system.delete_user_message(u_message)