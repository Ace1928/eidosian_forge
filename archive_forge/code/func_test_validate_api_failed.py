import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import workbooks
from mistralclient.tests.unit.v2 import base
def test_validate_api_failed(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, status_code=500)
    self.assertRaises(api_base.APIException, self.workbooks.validate, WB_DEF)
    last_request = self.requests_mock.last_request
    self.assertEqual(WB_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])