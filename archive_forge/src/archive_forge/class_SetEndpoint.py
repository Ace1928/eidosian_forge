import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetEndpoint(command.Command):
    _description = _('Set endpoint properties')

    def get_parser(self, prog_name):
        parser = super(SetEndpoint, self).get_parser(prog_name)
        parser.add_argument('endpoint', metavar='<endpoint-id>', help=_('Endpoint to modify (ID only)'))
        parser.add_argument('--region', metavar='<region-id>', help=_('New endpoint region ID'))
        parser.add_argument('--interface', metavar='<interface>', choices=['admin', 'public', 'internal'], help=_('New endpoint interface type (admin, public or internal)'))
        parser.add_argument('--url', metavar='<url>', help=_('New endpoint URL'))
        parser.add_argument('--service', metavar='<service>', help=_('New endpoint service (name or ID)'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', dest='enabled', action='store_true', help=_('Enable endpoint'))
        enable_group.add_argument('--disable', dest='disabled', action='store_true', help=_('Disable endpoint'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        endpoint = utils.find_resource(identity_client.endpoints, parsed_args.endpoint)
        service_id = None
        if parsed_args.service:
            service = common.find_service(identity_client, parsed_args.service)
            service_id = service.id
        enabled = None
        if parsed_args.enabled:
            enabled = True
        if parsed_args.disabled:
            enabled = False
        identity_client.endpoints.update(endpoint.id, service=service_id, url=parsed_args.url, interface=parsed_args.interface, region=parsed_args.region, enabled=enabled)