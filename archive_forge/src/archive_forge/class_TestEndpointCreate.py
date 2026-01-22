from openstackclient.identity.v2_0 import endpoint
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestEndpointCreate(TestEndpoint):

    def setUp(self):
        super(TestEndpointCreate, self).setUp()
        self.endpoints_mock.create.return_value = self.fake_endpoint
        self.services_mock.get.return_value = self.fake_service
        self.cmd = endpoint.CreateEndpoint(self.app, None)

    def test_endpoint_create(self):
        arglist = ['--publicurl', self.fake_endpoint.publicurl, '--internalurl', self.fake_endpoint.internalurl, '--adminurl', self.fake_endpoint.adminurl, '--region', self.fake_endpoint.region, self.fake_service.id]
        verifylist = [('adminurl', self.fake_endpoint.adminurl), ('internalurl', self.fake_endpoint.internalurl), ('publicurl', self.fake_endpoint.publicurl), ('region', self.fake_endpoint.region), ('service', self.fake_service.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.endpoints_mock.create.assert_called_with(self.fake_endpoint.region, self.fake_service.id, self.fake_endpoint.publicurl, self.fake_endpoint.adminurl, self.fake_endpoint.internalurl)
        collist = ('adminurl', 'id', 'internalurl', 'publicurl', 'region', 'service_id', 'service_name', 'service_type')
        self.assertEqual(collist, columns)
        datalist = (self.fake_endpoint.adminurl, self.fake_endpoint.id, self.fake_endpoint.internalurl, self.fake_endpoint.publicurl, self.fake_endpoint.region, self.fake_endpoint.service_id, self.fake_endpoint.service_name, self.fake_endpoint.service_type)
        self.assertEqual(datalist, data)