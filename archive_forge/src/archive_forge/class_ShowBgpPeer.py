from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class ShowBgpPeer(command.ShowOne):
    _description = _('Show information for a BGP peer')

    def get_parser(self, prog_name):
        parser = super(ShowBgpPeer, self).get_parser(prog_name)
        parser.add_argument('bgp_peer', metavar='<bgp-peer>', help=_('BGP peer to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgp_peer(parsed_args.bgp_peer, ignore_missing=False).id
        obj = client.get_bgp_peer(id)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)