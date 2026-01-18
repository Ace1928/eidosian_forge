import json
import mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def test_upload_documents_with_auth(self):
    layer1 = CloudSearchConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', sign_request=True)
    domain = Domain(layer1=layer1, data=json.loads(self.domain_status))
    document_service = domain.get_document_service()
    response = {'status': 'success', 'adds': 1, 'deletes': 0}
    document = {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']}
    self.set_http_response(status_code=200, body=json.dumps(response).encode('utf-8'))
    document_service.domain_connection = self.service_connection
    document_service.add('1234', document)
    resp = document_service.commit()
    headers = self.actual_request.headers
    self.assertIsNotNone(headers.get('Authorization'))