import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_queue_metadata_functional(self):
    queue = self.client.queue('meta-test', force_create=True)
    self.addCleanup(queue.delete)
    test_metadata = {'type': 'Bank Accounts', 'name': 'test1'}
    queue.metadata(new_meta=test_metadata)
    queue._metadata = None
    metadata = queue.metadata()
    expect_metadata = {'type': 'Bank Accounts', 'name': 'test1', '_max_messages_post_size': 262144, '_default_message_ttl': 3600, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None}
    self.assertEqual(expect_metadata, metadata)
    replace_add_metadata = {'type': 'test', 'name': 'test1', 'age': 13, '_default_message_ttl': 1000}
    queue.metadata(new_meta=replace_add_metadata)
    queue._metadata = None
    metadata = queue.metadata()
    expect_metadata = {'type': 'test', 'name': 'test1', 'age': 13, '_max_messages_post_size': 262144, '_default_message_ttl': 1000, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None}
    self.assertEqual(expect_metadata, metadata)
    replace_remove_add_metadata = {'name': 'test2', 'age': 13, 'fake': 'test_fake'}
    queue.metadata(new_meta=replace_remove_add_metadata)
    queue._metadata = None
    metadata = queue.metadata()
    expect_metadata = {'name': 'test2', 'age': 13, 'fake': 'test_fake', '_max_messages_post_size': 262144, '_default_message_ttl': 3600, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None}
    self.assertEqual(expect_metadata, metadata)
    replace_add_metadata = {'name': '', 'age': 13, 'fake': 'test_fake', 'empty_dict': {}}
    queue.metadata(new_meta=replace_add_metadata)
    queue._metadata = None
    metadata = queue.metadata()
    expect_metadata = {'name': '', 'age': 13, 'fake': 'test_fake', '_max_messages_post_size': 262144, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None, '_default_message_ttl': 3600, 'empty_dict': {}}
    self.assertEqual(expect_metadata, metadata)
    remove_all = {}
    queue.metadata(new_meta=remove_all)
    queue._metadata = None
    metadata = queue.metadata()
    expect_metadata = {'_max_messages_post_size': 262144, '_default_message_ttl': 3600, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None}
    self.assertEqual(expect_metadata, metadata)