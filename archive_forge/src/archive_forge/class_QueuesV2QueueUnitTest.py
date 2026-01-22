import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV2QueueUnitTest(QueuesV1_1QueueUnitTest):

    def test_message_get(self):
        pass

    def test_queue_subscriptions(self):
        result = {'subscriptions': [{'source': 'test', 'id': '1', 'subscriber': 'http://trigger.me', 'ttl': 3600, 'age': 1800, 'confirmed': False, 'options': {}}, {'source': 'test', 'id': '2', 'subscriber': 'http://trigger.you', 'ttl': 7200, 'age': 1800, 'confirmed': False, 'options': {}}]}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(result))
            send_method.return_value = resp
            subscriptions = self.queue.subscriptions()
            subscriber_list = [s.subscriber for s in list(subscriptions)]
            self.assertIn('http://trigger.me', subscriber_list)
            self.assertIn('http://trigger.you', subscriber_list)

    def test_queue_metadata(self):
        pass

    def test_queue_metadata_update(self):
        test_metadata = {'type': 'Bank Accounts', 'name': 'test1'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(test_metadata))
            send_method.return_value = resp
            metadata = self.queue.metadata(new_meta=test_metadata)
            self.assertEqual(test_metadata, metadata)
        new_metadata_replace = {'type': 'test', 'name': 'test1'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(new_metadata_replace))
            send_method.return_value = resp
            metadata = self.queue.metadata(new_meta=new_metadata_replace)
            expect_metadata = {'type': 'test', 'name': 'test1'}
            self.assertEqual(expect_metadata, metadata)
        remove_metadata = {'name': 'test1'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(remove_metadata))
            send_method.return_value = resp
            metadata = self.queue.metadata(new_meta=remove_metadata)
            expect_metadata = {'name': 'test1'}
            self.assertEqual(expect_metadata, metadata)

    def test_queue_purge(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.purge()

    def test_queue_purge_messages(self):
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            self.queue.purge(resource_types=['messages'])
            self.assertEqual({'resource_types': ['messages']}, json.loads(send_method.call_args[0][0].content))