import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api.v2 import workflows
from mistralclient.tests.unit.v2 import base
def test_update_with_id(self):
    self.requests_mock.put(self.TEST_URL + URL_TEMPLATE_NAME % '123', json=WORKFLOW)
    wf = self.workflows.update(WF_DEF, id='123')
    self.assertIsNotNone(wf)
    self.assertEqual(WF_DEF, wf.definition)
    last_request = self.requests_mock.last_request
    self.assertEqual('namespace=&scope=private', last_request.query)
    self.assertEqual(WF_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])