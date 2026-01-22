import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1QueueUnitTest(base.QueuesTestBase):

    def test_queue_metadata(self):
        test_metadata = {'type': 'Bank Accounts'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(test_metadata))
            send_method.return_value = resp
            metadata = self.queue.metadata(test_metadata)
            self.assertEqual(test_metadata, metadata)

    def test_queue_metadata_update(self):
        test_metadata = {'type': 'Bank Accounts'}
        new_meta = {'flavor': 'test'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(test_metadata))
            send_method.return_value = resp
            metadata = self.queue.metadata(test_metadata)
            self.assertEqual(test_metadata, metadata)
            metadata = self.queue.metadata(new_meta)
            self.assertEqual(new_meta, metadata)

    def test_queue_create(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.ensure_exists()

    def test_queue_valid_name(self):
        self.assertRaises(ValueError, self.client.queue, '')

    def test_queue_valid_name_with_pound(self):
        self.assertRaises(ValueError, self.client.queue, '123#456')

    def test_queue_valid_name_with_percent(self):
        self.assertRaises(ValueError, self.client.queue, '123%456')

    def test_queue_delete(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.delete()

    def test_queue_exists(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.exists()

    def test_queue_stats(self):
        result = {'messages': {'free': 146929, 'claimed': 2409, 'total': 149338, 'oldest': {'href': '/v1/queues/qq/messages/50b68a50d6f5b8c8a7c62b01', 'age': 63, 'created': '2013-08-12T20:44:55Z'}, 'newest': {'href': '/v1/queues/qq/messages/50b68a50d6f5b8c8a7c62b01', 'age': 12, 'created': '2013-08-12T20:45:46Z'}}}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(result))
            send_method.return_value = resp
            stats = self.queue.stats
            self.assertEqual(result, stats)

    def test_message_post(self):
        messages = [{'ttl': 30, 'body': 'Post It!'}]
        result = {'resources': ['/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01'], 'partial': False}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(result))
            send_method.return_value = resp
            posted = self.queue.post(messages)
            self.assertEqual(result, posted)

    def test_message_list(self):
        returned = {'links': [{'rel': 'next', 'href': '/v1/queues/fizbit/messages?marker=6244-244224-783'}], 'messages': [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(returned))
            send_method.return_value = resp
            msgs = self.queue.messages(limit=1)
            self.assertIsInstance(msgs, iterator._Iterator)

    def test_message_get(self):
        returned = {'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(returned))
            send_method.return_value = resp
            msgs = self.queue.message('50b68a50d6f5b8c8a7c62b01')
            self.assertIsInstance(msgs, message.Message)

    def test_message_get_many(self):
        returned = [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}, {'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b02', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(returned))
            send_method.return_value = resp
            msg = self.queue.messages('50b68a50d6f5b8c8a7c62b01', '50b68a50d6f5b8c8a7c62b02')
            self.assertIsInstance(msg, iterator._Iterator)

    def test_message_delete_many(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            rst = self.queue.delete_messages('50b68a50d6f5b8c8a7c62b01', '50b68a50d6f5b8c8a7c62b02')
            self.assertIsNone(rst)