import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_signed_url(self):
    queue = self.client.queue('test_queue')
    messages = [{'ttl': 300, 'body': 'Post It!'}]
    queue.post(messages)
    self.addCleanup(queue.delete)
    signature = queue.signed_url()
    opts = {'paths': signature['paths'], 'expires': signature['expires'], 'methods': signature['methods'], 'signature': signature['signature'], 'os_project_id': signature['project']}
    auth_opts = {'backend': 'signed-url', 'options': opts}
    conf = {'auth_opts': auth_opts}
    signed_client = client.Client(self.url, self.version, conf)
    queue = signed_client.queue('test_queue')
    [message] = list(queue.messages())
    self.assertEqual('Post It!', message.body)