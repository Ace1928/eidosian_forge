from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
class CloudSearchNonJsonTest(CloudSearchSearchBaseTest):
    response = b'<html><body><h1>500 Internal Server Error</h1></body></html>'
    response_status = 500
    content_type = 'text/xml'

    def test_response(self):
        search = SearchConnection(endpoint=HOSTNAME)
        with self.assertRaises(SearchServiceException):
            search.search(q='Test')