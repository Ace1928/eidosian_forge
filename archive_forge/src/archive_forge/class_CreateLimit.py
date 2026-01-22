import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class CreateLimit(command.ShowOne):
    _description = _('Create a limit')

    def get_parser(self, prog_name):
        parser = super(CreateLimit, self).get_parser(prog_name)
        parser.add_argument('--description', metavar='<description>', help=_('Description of the limit'))
        parser.add_argument('--region', metavar='<region>', help=_('Region for the limit to affect.'))
        parser.add_argument('--project', metavar='<project>', required=True, help=_('Project to associate the resource limit to'))
        parser.add_argument('--service', metavar='<service>', required=True, help=_('Service responsible for the resource to limit'))
        parser.add_argument('--resource-limit', metavar='<resource-limit>', required=True, type=int, help=_('The resource limit for the project to assume'))
        parser.add_argument('resource_name', metavar='<resource-name>', help=_('The name of the resource to limit'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        project = common_utils.find_project(identity_client, parsed_args.project)
        service = common_utils.find_service(identity_client, parsed_args.service)
        region = None
        if parsed_args.region:
            val = getattr(parsed_args, 'region', None)
            if 'None' not in val:
                region = common_utils.get_resource(identity_client.regions, parsed_args.region)
        limit = identity_client.limits.create(project, service, parsed_args.resource_name, parsed_args.resource_limit, description=parsed_args.description, region=region)
        limit._info.pop('links', None)
        return zip(*sorted(limit._info.items()))