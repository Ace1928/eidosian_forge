from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
class CloudSearchSearchBaseTest(unittest.TestCase):
    hits = [{'id': '12341', 'title': 'Document 1'}, {'id': '12342', 'title': 'Document 2'}, {'id': '12343', 'title': 'Document 3'}, {'id': '12344', 'title': 'Document 4'}, {'id': '12345', 'title': 'Document 5'}, {'id': '12346', 'title': 'Document 6'}, {'id': '12347', 'title': 'Document 7'}]
    content_type = 'text/xml'
    response_status = 200

    def get_args(self, requestline):
        _, request, _ = requestline.split(b' ')
        _, request = request.split(b'?', 1)
        args = six.moves.urllib.parse.parse_qs(request)
        return args

    def setUp(self):
        HTTPretty.enable()
        body = self.response
        if not isinstance(body, bytes):
            body = json.dumps(body).encode('utf-8')
        HTTPretty.register_uri(HTTPretty.GET, FULL_URL, body=body, content_type=self.content_type, status=self.response_status)

    def tearDown(self):
        HTTPretty.disable()