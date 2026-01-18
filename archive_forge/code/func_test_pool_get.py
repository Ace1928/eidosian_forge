import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_pool_get(self):
    pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
    pool = self.client.pool('FuncTestPool', **pool_data)
    resp_data = pool.get()
    self.addCleanup(pool.delete)
    self.assertEqual('FuncTestPool', resp_data['name'])
    self.assertEqual(10, resp_data['weight'])
    self.assertEqual('mongodb://127.0.0.1:27017', resp_data['uri'])