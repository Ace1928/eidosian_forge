from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class SetBgpPeer(command.Command):
    _description = _('Update a BGP peer')
    resource = constants.BGP_PEER

    def get_parser(self, prog_name):
        parser = super(SetBgpPeer, self).get_parser(prog_name)
        parser.add_argument('--name', help=_('Updated name of the BGP peer'))
        parser.add_argument('--password', metavar='<auth-password>', help=_('Updated authentication password'))
        parser.add_argument('bgp_peer', metavar='<bgp-peer>', help=_('BGP peer to update (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgp_peer(parsed_args.bgp_peer)['id']
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        client.update_bgp_peer(id, **attrs)