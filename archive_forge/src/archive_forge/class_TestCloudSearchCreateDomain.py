from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
class TestCloudSearchCreateDomain(AWSMockServiceTestCase):
    connection_class = Layer1

    def default_body(self):
        return b'\n<CreateDomainResponse xmlns="http://cloudsearch.amazonaws.com/doc/2011-02-01">\n  <CreateDomainResult>\n    <DomainStatus>\n      <SearchPartitionCount>0</SearchPartitionCount>\n      <SearchService>\n        <Arn>arn:aws:cs:us-east-1:1234567890:search/demo</Arn>\n        <Endpoint>search-demo-userdomain.us-east-1.cloudsearch.amazonaws.com</Endpoint>\n      </SearchService>\n      <NumSearchableDocs>0</NumSearchableDocs>\n      <Created>true</Created>\n      <DomainId>1234567890/demo</DomainId>\n      <Processing>false</Processing>\n      <SearchInstanceCount>0</SearchInstanceCount>\n      <DomainName>demo</DomainName>\n      <RequiresIndexDocuments>false</RequiresIndexDocuments>\n      <Deleted>false</Deleted>\n      <DocService>\n        <Arn>arn:aws:cs:us-east-1:1234567890:doc/demo</Arn>\n        <Endpoint>doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com</Endpoint>\n      </DocService>\n    </DomainStatus>\n  </CreateDomainResult>\n  <ResponseMetadata>\n    <RequestId>00000000-0000-0000-0000-000000000000</RequestId>\n  </ResponseMetadata>\n</CreateDomainResponse>\n'

    def test_create_domain(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_domain('demo')
        self.assert_request_parameters({'Action': 'CreateDomain', 'DomainName': 'demo', 'Version': '2011-02-01'})

    def test_cloudsearch_connect_result_endpoints(self):
        """Check that endpoints & ARNs are correctly returned from AWS"""
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_domain('demo')
        domain = Domain(self, api_response)
        self.assertEqual(domain.doc_service_arn, 'arn:aws:cs:us-east-1:1234567890:doc/demo')
        self.assertEqual(domain.doc_service_endpoint, 'doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        self.assertEqual(domain.search_service_arn, 'arn:aws:cs:us-east-1:1234567890:search/demo')
        self.assertEqual(domain.search_service_endpoint, 'search-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')

    def test_cloudsearch_connect_result_statuses(self):
        """Check that domain statuses are correctly returned from AWS"""
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_domain('demo')
        domain = Domain(self, api_response)
        self.assertEqual(domain.created, True)
        self.assertEqual(domain.processing, False)
        self.assertEqual(domain.requires_index_documents, False)
        self.assertEqual(domain.deleted, False)

    def test_cloudsearch_connect_result_details(self):
        """Check that the domain information is correctly returned from AWS"""
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_domain('demo')
        domain = Domain(self, api_response)
        self.assertEqual(domain.id, '1234567890/demo')
        self.assertEqual(domain.name, 'demo')

    def test_cloudsearch_documentservice_creation(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_domain('demo')
        domain = Domain(self, api_response)
        document = domain.get_document_service()
        self.assertEqual(document.endpoint, 'doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')

    def test_cloudsearch_searchservice_creation(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_domain('demo')
        domain = Domain(self, api_response)
        search = domain.get_search_service()
        self.assertEqual(search.endpoint, 'search-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')