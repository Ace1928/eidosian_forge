import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowServiceProvider(command.ShowOne):
    _description = _('Display service provider details')

    def get_parser(self, prog_name):
        parser = super(ShowServiceProvider, self).get_parser(prog_name)
        parser.add_argument('service_provider', metavar='<service-provider>', help=_('Service provider to display'))
        return parser

    def take_action(self, parsed_args):
        service_client = self.app.client_manager.identity
        service_provider = utils.find_resource(service_client.federation.service_providers, parsed_args.service_provider, id=parsed_args.service_provider)
        service_provider._info.pop('links', None)
        return zip(*sorted(service_provider._info.items()))