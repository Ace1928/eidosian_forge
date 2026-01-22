import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn import constants
class SetBgpvpnResAssoc(command.Command):
    """Set BGP VPN resource association properties"""
    _action = 'set'

    def get_parser(self, prog_name):
        parser = super(SetBgpvpnResAssoc, self).get_parser(prog_name)
        parser.add_argument('resource_association_id', metavar='<%s association ID>' % self._assoc_res_name, help=_('%s association ID to update') % self._assoc_res_name.capitalize())
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN the %s association belongs to (name or ID)') % self._assoc_res_name)
        get_common_parser = getattr(self, '_get_common_parser', None)
        if callable(get_common_parser):
            get_common_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        bgpvpn = client.find_bgpvpn(parsed_args.bgpvpn)
        arg2body = getattr(self, '_args2body', None)
        if callable(arg2body):
            body = arg2body(bgpvpn['id'], parsed_args)
            if self._assoc_res_name == constants.NETWORK_ASSOC:
                client.update_bgpvpn_network_association(bgpvpn['id'], parsed_args.resource_association_id, **body)
            elif self._assoc_res_name == constants.PORT_ASSOCS:
                client.update_bgpvpn_port_association(bgpvpn['id'], parsed_args.resource_association_id, **body)
            else:
                client.update_bgpvpn_router_association(bgpvpn['id'], parsed_args.resource_association_id, **body)