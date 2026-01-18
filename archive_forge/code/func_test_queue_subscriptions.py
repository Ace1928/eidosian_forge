import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_queue_subscriptions(self):
    queue_name = 'test_queue'
    queue = self.client.queue(queue_name, force_create=True)
    self.addCleanup(queue.delete)
    queue._get_transport = mock.Mock(return_value=self.transport)
    subscription.Subscription(self.client, queue_name, subscriber='http://trigger.me')
    subscription.Subscription(self.client, queue_name, subscriber='http://trigger.you')
    get_subscriptions = queue.subscriptions()
    self.assertIsInstance(get_subscriptions, iterator._Iterator)
    self.assertEqual(2, len(list(get_subscriptions)))