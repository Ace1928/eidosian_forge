from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_show_with_domain(self):
    project = identity_fakes.FakeProject.create_one_project({'name': self.project.name})
    self.app.client_manager.identity.tokens.get_token_data.return_value = {'token': {'project': {'domain': {'id': self.project.domain_id}, 'name': self.project.name, 'id': self.project.id}}}
    identity_client = self.app.client_manager.identity
    arglist = ['--domain', self.domain.id, project.name]
    verifylist = [('domain', self.domain.id), ('project', project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    project_str = common._get_token_resource(identity_client, 'project', parsed_args.project, parsed_args.domain)
    self.assertEqual(self.project.id, project_str)
    arglist = ['--domain', project.domain_id, project.name]
    verifylist = [('domain', project.domain_id), ('project', project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    project_str = common._get_token_resource(identity_client, 'project', parsed_args.project, parsed_args.domain)
    self.assertEqual(project.name, project_str)