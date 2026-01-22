import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1_1QueueUnitTest(QueuesV1QueueUnitTest):

    def test_message_pop(self):
        returned = [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}, {'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b02', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(returned))
            send_method.return_value = resp
            msg = self.queue.pop(count=2)
            self.assertIsInstance(msg, iterator._Iterator)

    def test_queue_metadata(self):
        test_metadata = {'type': 'Bank Accounts'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(test_metadata))
            send_method.return_value = resp
            self.assertRaises(RuntimeError, self.queue.metadata, test_metadata)

    def test_queue_metadata_update(self):
        pass