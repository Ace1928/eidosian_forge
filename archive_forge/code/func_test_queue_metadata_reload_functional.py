import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_queue_metadata_reload_functional(self):
    test_metadata = {'type': 'Bank Accounts', 'name': 'test1'}
    queue = self.client.queue('meta-test', force_create=True)
    self.addCleanup(queue.delete)
    queue.metadata(new_meta=test_metadata)
    queue._metadata = 'test'
    expect_metadata = {'type': 'Bank Accounts', 'name': 'test1', '_max_messages_post_size': 262144, '_default_message_ttl': 3600, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None}
    metadata = queue.metadata(force_reload=True)
    self.assertEqual(expect_metadata, metadata)