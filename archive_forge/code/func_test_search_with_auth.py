import json
import mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def test_search_with_auth(self):
    layer1 = CloudSearchConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', sign_request=True)
    domain = Domain(layer1=layer1, data=json.loads(self.domain_status))
    search_service = domain.get_search_service()
    response = {'rank': '-text_relevance', 'match-expr': 'Test', 'hits': {'found': 30, 'start': 0, 'hit': {'id': '12341', 'fields': {'title': 'Document 1', 'rank': 1}}}, 'status': {'rid': 'b7c167f6c2da6d93531b9a7b314ad030b3a74803b4b7797edb905ba5a6a08', 'time-ms': 2, 'cpu-time-ms': 0}}
    self.set_http_response(status_code=200, body=json.dumps(response).encode('utf-8'))
    search_service.domain_connection = self.service_connection
    resp = search_service.search()
    headers = self.actual_request.headers
    self.assertIsNotNone(headers.get('Authorization'))