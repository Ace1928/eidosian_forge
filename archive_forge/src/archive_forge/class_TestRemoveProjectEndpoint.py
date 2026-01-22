from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRemoveProjectEndpoint(TestEndpoint):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    service = identity_fakes.FakeService.create_one_service()
    endpoint = identity_fakes.FakeEndpoint.create_one_endpoint(attrs={'service_id': service.id})

    def setUp(self):
        super(TestRemoveProjectEndpoint, self).setUp()
        self.endpoints_mock.get.return_value = self.endpoint
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.ep_filter_mock.delete.return_value = None
        self.cmd = endpoint.RemoveProjectFromEndpoint(self.app, None)

    def test_remove_project_endpoint_no_options(self):
        arglist = [self.endpoint.id, self.project.id]
        verifylist = [('endpoint', self.endpoint.id), ('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.ep_filter_mock.delete_endpoint_from_project.assert_called_with(project=self.project.id, endpoint=self.endpoint.id)
        self.assertIsNone(result)

    def test_remove_project_endpoint_with_options(self):
        arglist = [self.endpoint.id, self.project.id, '--project-domain', self.domain.id]
        verifylist = [('endpoint', self.endpoint.id), ('project', self.project.id), ('project_domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.ep_filter_mock.delete_endpoint_from_project.assert_called_with(project=self.project.id, endpoint=self.endpoint.id)
        self.assertIsNone(result)