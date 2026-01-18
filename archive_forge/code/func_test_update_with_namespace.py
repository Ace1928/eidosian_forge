import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import actions
from mistralclient.tests.unit.v2 import base
def test_update_with_namespace(self):
    self.requests_mock.put(self.TEST_URL + URL_TEMPLATE, json={'actions': [ACTION]})
    actions = self.actions.update(ACTION_DEF, namespace='test_namespace')
    self.assertIsNotNone(actions)
    self.assertEqual(ACTION_DEF, actions[0].definition)
    last_request = self.requests_mock.last_request
    self.assertEqual('scope=private&namespace=test_namespace', last_request.query)
    self.assertEqual('text/plain', last_request.headers['content-type'])
    self.assertEqual(ACTION_DEF, last_request.text)