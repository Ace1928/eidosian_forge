import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestLimitCreate(TestLimit):

    def setUp(self):
        super(TestLimitCreate, self).setUp()
        self.service = fakes.FakeResource(None, copy.deepcopy(identity_fakes.SERVICE), loaded=True)
        self.services_mock.get.return_value = self.service
        self.project = fakes.FakeResource(None, copy.deepcopy(identity_fakes.PROJECT), loaded=True)
        self.projects_mock.get.return_value = self.project
        self.region = fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGION), loaded=True)
        self.regions_mock.get.return_value = self.region
        self.cmd = limit.CreateLimit(self.app, None)

    def test_limit_create_without_options(self):
        self.limit_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.LIMIT), loaded=True)
        resource_limit = 15
        arglist = ['--project', identity_fakes.project_id, '--service', identity_fakes.service_id, '--resource-limit', str(resource_limit), identity_fakes.limit_resource_name]
        verifylist = [('project', identity_fakes.project_id), ('service', identity_fakes.service_id), ('resource_name', identity_fakes.limit_resource_name), ('resource_limit', resource_limit)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'region': None}
        self.limit_mock.create.assert_called_with(self.project, self.service, identity_fakes.limit_resource_name, resource_limit, **kwargs)
        collist = ('description', 'id', 'project_id', 'region_id', 'resource_limit', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (None, identity_fakes.limit_id, identity_fakes.project_id, None, resource_limit, identity_fakes.limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)

    def test_limit_create_with_options(self):
        self.limit_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.LIMIT_OPTIONS), loaded=True)
        resource_limit = 15
        arglist = ['--project', identity_fakes.project_id, '--service', identity_fakes.service_id, '--resource-limit', str(resource_limit), '--region', identity_fakes.region_id, '--description', identity_fakes.limit_description, identity_fakes.limit_resource_name]
        verifylist = [('project', identity_fakes.project_id), ('service', identity_fakes.service_id), ('resource_name', identity_fakes.limit_resource_name), ('resource_limit', resource_limit), ('region', identity_fakes.region_id), ('description', identity_fakes.limit_description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': identity_fakes.limit_description, 'region': self.region}
        self.limit_mock.create.assert_called_with(self.project, self.service, identity_fakes.limit_resource_name, resource_limit, **kwargs)
        collist = ('description', 'id', 'project_id', 'region_id', 'resource_limit', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.limit_description, identity_fakes.limit_id, identity_fakes.project_id, identity_fakes.region_id, resource_limit, identity_fakes.limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)