import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_subscription_update(self):
    sub = self.client.subscription(self.queue_name, auto_create=False, **{'id': self.subscription_1.id})
    data = {'subscriber': 'http://trigger.ok', 'ttl': 1000}
    sub.update(data)
    self.assertEqual('http://trigger.ok', sub.subscriber)
    self.assertEqual(1000, sub.ttl)