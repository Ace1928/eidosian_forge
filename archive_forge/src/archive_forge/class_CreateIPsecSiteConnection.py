import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class CreateIPsecSiteConnection(IPsecSiteConnectionMixin, neutronv20.CreateCommand):
    """Create an IPsec site connection."""
    resource = 'ipsec_site_connection'

    def add_known_arguments(self, parser):
        parser.add_argument('--admin-state-down', default=True, action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--vpnservice-id', metavar='VPNSERVICE', required=True, help=_('VPN service instance ID associated with this connection.'))
        parser.add_argument('--ikepolicy-id', metavar='IKEPOLICY', required=True, help=_('IKE policy ID associated with this connection.'))
        parser.add_argument('--ipsecpolicy-id', metavar='IPSECPOLICY', required=True, help=_('IPsec policy ID associated with this connection.'))
        super(CreateIPsecSiteConnection, self).add_known_arguments(parser)

    def args2body(self, parsed_args):
        _vpnservice_id = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'vpnservice', parsed_args.vpnservice_id)
        _ikepolicy_id = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'ikepolicy', parsed_args.ikepolicy_id)
        _ipsecpolicy_id = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'ipsecpolicy', parsed_args.ipsecpolicy_id)
        body = {'vpnservice_id': _vpnservice_id, 'ikepolicy_id': _ikepolicy_id, 'ipsecpolicy_id': _ipsecpolicy_id, 'admin_state_up': parsed_args.admin_state_down}
        if parsed_args.tenant_id:
            body['tenant_id'] = parsed_args.tenant_id
        if bool(parsed_args.local_ep_group) != bool(parsed_args.peer_ep_group):
            message = _('You must specify both local and peer endpoint groups.')
            raise exceptions.CommandError(message)
        if not parsed_args.peer_cidrs and (not parsed_args.local_ep_group):
            message = _('You must specify endpoint groups or peer CIDR(s).')
            raise exceptions.CommandError(message)
        return super(CreateIPsecSiteConnection, self).args2body(parsed_args, body)