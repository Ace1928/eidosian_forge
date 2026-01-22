import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class CreateRegisteredLimit(command.ShowOne):
    _description = _('Create a registered limit')

    def get_parser(self, prog_name):
        parser = super(CreateRegisteredLimit, self).get_parser(prog_name)
        parser.add_argument('--description', metavar='<description>', help=_('Description of the registered limit'))
        parser.add_argument('--region', metavar='<region>', help=_('Region for the registered limit to affect'))
        parser.add_argument('--service', metavar='<service>', required=True, help=_('Service responsible for the resource to limit (required)'))
        parser.add_argument('--default-limit', type=int, metavar='<default-limit>', required=True, help=_('The default limit for the resources to assume (required)'))
        parser.add_argument('resource_name', metavar='<resource-name>', help=_('The name of the resource to limit'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        service = utils.find_resource(identity_client.services, parsed_args.service)
        region = None
        if parsed_args.region:
            val = getattr(parsed_args, 'region', None)
            if 'None' not in val:
                region = common_utils.get_resource(identity_client.regions, parsed_args.region)
        registered_limit = identity_client.registered_limits.create(service, parsed_args.resource_name, parsed_args.default_limit, description=parsed_args.description, region=region)
        registered_limit._info.pop('links', None)
        return zip(*sorted(registered_limit._info.items()))