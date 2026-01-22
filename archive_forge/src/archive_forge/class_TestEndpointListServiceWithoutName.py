from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestEndpointListServiceWithoutName(TestEndpointList):
    service = identity_fakes.FakeService.create_one_service(attrs={'service_name': ''})
    endpoint = identity_fakes.FakeEndpoint.create_one_endpoint(attrs={'service_id': service.id})

    def setUp(self):
        super(TestEndpointList, self).setUp()
        self.endpoints_mock.list.return_value = [self.endpoint]
        self.services_mock.get.return_value = self.service
        self.services_mock.list.return_value = [self.service]
        self.cmd = endpoint.ListEndpoint(self.app, None)