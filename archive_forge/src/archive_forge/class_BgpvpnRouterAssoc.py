from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
class BgpvpnRouterAssoc(object):
    _assoc_res_name = constants.ROUTER_RESOURCE_NAME
    _resource = constants.ROUTER_ASSOC
    _resource_plural = constants.ROUTER_ASSOCS
    _attr_map = (('id', 'ID', column_util.LIST_BOTH), ('tenant_id', 'Project', column_util.LIST_LONG_ONLY), ('%s_id' % _assoc_res_name, '%s ID' % _assoc_res_name.capitalize(), column_util.LIST_BOTH), ('advertise_extra_routes', 'Advertise extra routes', column_util.LIST_LONG_ONLY))
    _formatters = {}

    def _get_common_parser(self, parser):
        """Adds to parser arguments common to create, set and unset commands.

        :params ArgumentParser parser: argparse object contains all command's
                                       arguments
        """
        ADVERTISE_ROUTES = _('Routes will be advertised to the BGP VPN%s') % (_(' (default)') if self._action == 'create' else '')
        NOT_ADVERTISE_ROUTES = _('Routes from the router will not be advertised to the BGP VPN')
        group_advertise_extra_routes = parser.add_mutually_exclusive_group()
        group_advertise_extra_routes.add_argument('--advertise_extra_routes', action='store_true', help=NOT_ADVERTISE_ROUTES if self._action == 'unset' else ADVERTISE_ROUTES)
        group_advertise_extra_routes.add_argument('--no-advertise_extra_routes', action='store_true', help=ADVERTISE_ROUTES if self._action == 'unset' else NOT_ADVERTISE_ROUTES)

    def _args2body(self, _, args):
        attrs = {'advertise_extra_routes': False}
        if args.advertise_extra_routes:
            attrs['advertise_extra_routes'] = self._action != 'unset'
        elif args.no_advertise_extra_routes:
            attrs['advertise_extra_routes'] = self._action == 'unset'
        return attrs