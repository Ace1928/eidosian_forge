import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn import constants
class DeleteBgpvpnResAssoc(command.Command):
    """Remove a BGP VPN resource association(s) for a given BGP VPN"""

    def get_parser(self, prog_name):
        parser = super(DeleteBgpvpnResAssoc, self).get_parser(prog_name)
        parser.add_argument('resource_association_ids', metavar='<%s association ID>' % self._assoc_res_name, nargs='+', help=_('%s association ID(s) to remove') % self._assoc_res_name.capitalize())
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN the %s association belongs to (name or ID)') % self._assoc_res_name)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        bgpvpn = client.find_bgpvpn(parsed_args.bgpvpn)
        fails = 0
        for id in parsed_args.resource_association_ids:
            try:
                if self._assoc_res_name == constants.NETWORK_ASSOC:
                    client.delete_bgpvpn_network_association(bgpvpn['id'], id)
                elif self._assoc_res_name == constants.PORT_ASSOCS:
                    client.delete_bgpvpn_port_association(bgpvpn['id'], id)
                else:
                    client.delete_bgpvpn_router_association(bgpvpn['id'], id)
                LOG.warning('%(assoc_res_name)s association %(id)s deleted', {'assoc_res_name': self._assoc_res_name.capitalize(), 'id': id})
            except Exception as e:
                fails += 1
                LOG.error("Failed to delete %(assoc_res_name)s association with ID '%(id)s': %(e)s", {'assoc_res_name': self._assoc_res_name, 'id': id, 'e': e})
        if fails > 0:
            msg = _('Failed to delete %(fails)s of %(total)s %(assoc_res_name)s BGP VPN association(s).') % {'fails': fails, 'total': len(parsed_args.resource_association_ids), 'assoc_res_name': self._assoc_res_name}
            raise exceptions.CommandError(msg)