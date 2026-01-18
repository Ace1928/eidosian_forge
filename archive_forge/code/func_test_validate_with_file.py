import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import workbooks
from mistralclient.tests.unit.v2 import base
def test_validate_with_file(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, json={'valid': True})
    path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wb_v2.yaml')
    result = self.workbooks.validate(path)
    self.assertIsNotNone(result)
    self.assertIn('valid', result)
    self.assertTrue(result['valid'])
    last_request = self.requests_mock.last_request
    self.assertEqual(WB_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])