import logging
from operator import xor
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class ShowShareNetworkSubnet(command.ShowOne):
    """Show share network subnet."""
    _description = _('Show share network subnet')

    def get_parser(self, prog_name):
        parser = super(ShowShareNetworkSubnet, self).get_parser(prog_name)
        parser.add_argument('share_network', metavar='<share-network>', help=_('Share network name or ID.'))
        parser.add_argument('share_network_subnet', metavar='<share-network-subnet>', help=_('ID of share network subnet to show.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_network_id = oscutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        share_network_subnet = share_client.share_network_subnets.get(share_network_id, parsed_args.share_network_subnet)
        data = share_network_subnet._info
        data.update({'properties': format_columns.DictColumn(data.pop('metadata', {}))})
        return self.dict2columns(data)