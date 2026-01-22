import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV2QueueFunctionalTest(QueuesV1_1QueueFunctionalTest):

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

    def test_queue_metadata_reload_functional(self):
        test_metadata = {'type': 'Bank Accounts', 'name': 'test1'}
        queue = self.client.queue('meta-test', force_create=True)
        self.addCleanup(queue.delete)
        queue.metadata(new_meta=test_metadata)
        queue._metadata = 'test'
        expect_metadata = {'type': 'Bank Accounts', 'name': 'test1', '_max_messages_post_size': 262144, '_default_message_ttl': 3600, '_default_message_delay': 0, '_dead_letter_queue': None, '_dead_letter_queue_messages_ttl': None, '_max_claim_count': None}
        metadata = queue.metadata(force_reload=True)
        self.assertEqual(expect_metadata, metadata)

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