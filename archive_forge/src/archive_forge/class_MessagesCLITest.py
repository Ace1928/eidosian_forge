from manilaclient.tests.functional.osc import base
class MessagesCLITest(base.OSCClientTestBase):

    def setUp(self):
        super(MessagesCLITest, self).setUp()
        st = self.create_share_type(extra_specs={'fake_key': 'fake_value'})
        self.share = self.create_share(share_type=st['name'], wait_for_status='error', client=self.user_client)

    def test_list_and_show_messages(self):
        messages = self.listing_result('share', 'message list', client=self.user_client)
        self.assertTrue(len(messages) > 0)
        self.assertTableStruct(messages, ['ID', 'Resource Type', 'Resource ID', 'Action ID', 'User Message', 'Detail ID', 'Created At'])
        message = [msg for msg in messages if msg['Resource ID'] == self.share['id']]
        self.assertEqual(1, len(message))
        show_message = self.dict_result('share', f'message show {message[0]['ID']}')
        self.addCleanup(self.openstack, f'share message delete {show_message['id']}')
        self.assertEqual(message[0]['ID'], show_message['id'])
        expected_keys = ('id', 'action_id', 'resource_id', 'detail_id', 'resource_type', 'created_at', 'expires_at', 'message_level', 'user_message', 'request_id')
        for key in expected_keys:
            self.assertIn(key, show_message)
        since = show_message['created_at']
        before = show_message['expires_at']
        filtered_messages = self.listing_result('share', f'message list --since {since} --before {before}', client=self.user_client)
        self.assertTrue(len(filtered_messages) > 0)
        self.assertIn(show_message['id'], [m['ID'] for m in filtered_messages])
        filtered_messages = self.listing_result('share', f'message list --message-level {show_message['message_level']}', client=self.user_client)
        self.assertTrue(len(filtered_messages) > 0)
        self.assertIn(show_message['id'], [m['ID'] for m in filtered_messages])
        filtered_messages = self.listing_result('share', f'message list --resource-id {self.share['id']}', client=self.user_client)
        self.assertEqual(1, len(filtered_messages))
        self.assertEqual(show_message['resource_id'], self.share['id'])

    def test_delete_message(self):
        messages = self.listing_result('share', 'message list', client=self.user_client)
        message = [msg for msg in messages if msg['Resource ID'] == self.share['id']]
        self.assertEqual(1, len(message))
        message = message[0]
        self.openstack(f'share message delete {message['ID']}')
        messages = self.listing_result('share', 'message list', client=self.user_client)
        messages = [msg for msg in messages if msg['ID'] == message['ID']]
        self.assertEqual(0, len(messages))