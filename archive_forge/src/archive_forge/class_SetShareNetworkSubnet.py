import logging
from operator import xor
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class SetShareNetworkSubnet(command.Command):
    """Set share network subnet properties."""
    _description = _('Set share network subnet properties')

    def get_parser(self, prog_name):
        parser = super(SetShareNetworkSubnet, self).get_parser(prog_name)
        parser.add_argument('share_network', metavar='<share-network>', help=_('Share network name or ID.'))
        parser.add_argument('share_network_subnet', metavar='<share-network-subnet>', help=_('ID of share network subnet to set a property.'))
        parser.add_argument('--property', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set a property to this share network subnet (repeat option to set multiple properties). Available only for microversion >= 2.78.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if parsed_args.property and share_client.api_version < api_versions.APIVersion('2.78'):
            raise exceptions.CommandError('Property can be specified only with manila API version >= 2.78.')
        share_network_id = oscutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        if parsed_args.property:
            try:
                share_client.share_network_subnets.set_metadata(share_network_id, parsed_args.property, subresource=parsed_args.share_network_subnet)
            except Exception as e:
                raise exceptions.CommandError(_("Failed to set subnet property '%(properties)s': %(e)s") % {'properties': parsed_args.property, 'e': e})