import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1QueueFunctionalTest(base.QueuesTestBase):

    def test_queue_create_functional(self):
        queue = self.client.queue('nonono')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        self.assertTrue(queue.exists())

    def test_queue_delete_functional(self):
        queue = self.client.queue('nonono')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        self.assertTrue(queue.exists())
        queue.delete()
        self.assertFalse(queue.exists())

    def test_queue_exists_functional(self):
        queue = self.client.queue('404', auto_create=False)
        queue._get_transport = mock.Mock(return_value=self.transport)
        self.assertFalse(queue.exists())

    def test_queue_stats_functional(self):
        messages = [{'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}]
        queue = self.client.queue('nonono')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        queue.post(messages)
        stats = queue.stats
        self.assertEqual(3, stats['messages']['free'])

    def test_queue_metadata_functional(self):
        test_metadata = {'type': 'Bank Accounts'}
        queue = self.client.queue('meta-test')
        queue.metadata(test_metadata)
        queue._metadata = None
        metadata = queue.metadata()
        self.assertEqual(test_metadata['type'], metadata['type'])

    def test_queue_metadata_reload_functional(self):
        test_metadata = {'type': 'Bank Accounts'}
        queue = self.client.queue('meta-test')
        queue.metadata(test_metadata)
        queue._metadata = 'test'
        metadata = queue.metadata(force_reload=True)
        self.assertEqual(test_metadata['type'], metadata['type'])

    def test_message_post_functional(self):
        messages = [{'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}]
        queue = self.client.queue('nonono')
        queue._get_transport = mock.Mock(return_value=self.transport)
        result = queue.post(messages)
        self.assertIn('resources', result)
        self.assertEqual(3, len(result['resources']))

    def test_message_list_functional(self):
        queue = self.client.queue('test_queue')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}]
        queue.post(messages)
        messages = queue.messages()
        self.assertIsInstance(messages, iterator._Iterator)
        self.assertGreaterEqual(len(list(messages)), 0)

    def test_message_list_echo_functional(self):
        queue = self.client.queue('test_queue')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
        queue.post(messages)
        messages = queue.messages(echo=True)
        self.assertIsInstance(messages, iterator._Iterator)
        self.assertGreaterEqual(len(list(messages)), 3)

    def test_message_get_functional(self):
        queue = self.client.queue('test_queue')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
        res = queue.post(messages)['resources']
        msg_id = res[0].split('/')[-1]
        msg = queue.message(msg_id)
        self.assertIsInstance(msg, message.Message)
        self.assertEqual(res[0], msg.href)

    def test_message_get_many_functional(self):
        queue = self.client.queue('test_queue')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
        res = queue.post(messages)['resources']
        msgs_id = [ref.split('/')[-1] for ref in res]
        messages = queue.messages(*msgs_id)
        self.assertIsInstance(messages, iterator._Iterator)
        messages = list(messages)
        length = len(messages)
        if length == 3:
            bodies = set((message.body for message in messages))
            self.assertEqual(set(['Post It 1!', 'Post It 2!', 'Post It 3!']), bodies)
        elif length == 1:
            pass
        else:
            self.fail("Wrong number of messages: '%d'" % length)

    def test_message_delete_many_functional(self):
        queue = self.client.queue('test_queue')
        self.addCleanup(queue.delete)
        queue._get_transport = mock.Mock(return_value=self.transport)
        messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}]
        res = queue.post(messages)['resources']
        msgs_id = [ref.split('/')[-1] for ref in res]
        queue.delete_messages(*msgs_id)
        messages = queue.messages()
        self.assertEqual(0, len(list(messages)))