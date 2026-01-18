import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_subscription_delete(self):
    self.subscription_1.delete()
    subscription_data = {'id': self.subscription_1.id}
    self.assertRaises(errors.ResourceNotFound, self.client.subscription, self.queue_name, **subscription_data)