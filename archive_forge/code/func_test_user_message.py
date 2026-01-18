from openstack.shared_file_system.v2 import user_message
from openstack.tests.unit import base
def test_user_message(self):
    messages = user_message.UserMessage(**EXAMPLE)
    self.assertEqual(EXAMPLE['id'], messages.id)
    self.assertEqual(EXAMPLE['resource_id'], messages.resource_id)
    self.assertEqual(EXAMPLE['message_level'], messages.message_level)
    self.assertEqual(EXAMPLE['user_message'], messages.user_message)
    self.assertEqual(EXAMPLE['expires_at'], messages.expires_at)
    self.assertEqual(EXAMPLE['detail_id'], messages.detail_id)
    self.assertEqual(EXAMPLE['created_at'], messages.created_at)
    self.assertEqual(EXAMPLE['request_id'], messages.request_id)
    self.assertEqual(EXAMPLE['project_id'], messages.project_id)
    self.assertEqual(EXAMPLE['resource_type'], messages.resource_type)
    self.assertEqual(EXAMPLE['action_id'], messages.action_id)