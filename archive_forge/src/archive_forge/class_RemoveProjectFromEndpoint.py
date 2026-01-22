import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class RemoveProjectFromEndpoint(command.Command):
    _description = _('Dissociate a project from an endpoint')

    def get_parser(self, prog_name):
        parser = super(RemoveProjectFromEndpoint, self).get_parser(prog_name)
        parser.add_argument('endpoint', metavar='<endpoint>', help=_('Endpoint to dissociate from specified project (name or ID)'))
        parser.add_argument('project', metavar='<project>', help=_('Project to dissociate from specified endpoint name or ID)'))
        common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.identity
        endpoint = utils.find_resource(client.endpoints, parsed_args.endpoint)
        project = common.find_project(client, parsed_args.project, parsed_args.project_domain)
        client.endpoint_filter.delete_endpoint_from_project(project=project.id, endpoint=endpoint.id)