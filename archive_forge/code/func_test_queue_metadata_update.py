import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
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