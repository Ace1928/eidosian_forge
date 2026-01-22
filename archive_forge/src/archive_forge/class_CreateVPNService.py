from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class CreateVPNService(neutronv20.CreateCommand):
    """Create a VPN service."""
    resource = 'vpnservice'

    def add_known_arguments(self, parser):
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('router', metavar='ROUTER', help=_('Router unique identifier for the VPN service.'))
        parser.add_argument('subnet', nargs='?', metavar='SUBNET', help=_('[DEPRECATED in Mitaka] Unique identifier for the local private subnet.'))
        add_common_args(parser)

    def args2body(self, parsed_args):
        if parsed_args.subnet:
            _subnet_id = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'subnet', parsed_args.subnet)
        else:
            _subnet_id = None
        _router_id = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'router', parsed_args.router)
        body = {'subnet_id': _subnet_id, 'router_id': _router_id, 'admin_state_up': parsed_args.admin_state}
        neutronv20.update_dict(parsed_args, body, ['tenant_id'])
        common_args2body(parsed_args, body)
        return {self.resource: body}