import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class IPsecSiteConnectionMixin(object):

    def add_known_arguments(self, parser, is_create=True):
        parser.add_argument('--name', help=_('Set friendly name for the connection.'))
        parser.add_argument('--description', help=_('Set a description for the connection.'))
        parser.add_argument('--dpd', metavar='action=ACTION,interval=INTERVAL,timeout=TIMEOUT', type=utils.str2dict_type(optional_keys=['action', 'interval', 'timeout']), help=vpn_utils.dpd_help('IPsec connection.'))
        parser.add_argument('--local-ep-group', help=_('Local endpoint group ID/name with subnet(s) for IPSec connection.'))
        parser.add_argument('--peer-ep-group', help=_('Peer endpoint group ID/name with CIDR(s) for IPSec connection.'))
        parser.add_argument('--peer-cidr', action='append', dest='peer_cidrs', help=_('[DEPRECATED in Mitaka] Remote subnet(s) in CIDR format. Cannot be specified when using endpoint groups. Only applicable, if subnet provided for VPN service.'))
        parser.add_argument('--peer-id', required=is_create, help=_('Peer router identity for authentication. Can be IPv4/IPv6 address, e-mail address, key id, or FQDN.'))
        parser.add_argument('--peer-address', required=is_create, help=_('Peer gateway public IPv4/IPv6 address or FQDN.'))
        parser.add_argument('--psk', required=is_create, help=_('Pre-shared key string.'))
        parser.add_argument('--mtu', default='1500' if is_create else argparse.SUPPRESS, help=_('MTU size for the connection, default:1500.'))
        parser.add_argument('--initiator', default='bi-directional' if is_create else argparse.SUPPRESS, choices=['bi-directional', 'response-only'], help=_('Initiator state in lowercase, default:bi-directional'))

    def args2body(self, parsed_args, body=None):
        """Add in conditional args and then return all conn info."""
        if body is None:
            body = {}
        if parsed_args.dpd:
            vpn_utils.validate_dpd_dict(parsed_args.dpd)
            body['dpd'] = parsed_args.dpd
        if parsed_args.local_ep_group:
            _local_epg = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'endpoint_group', parsed_args.local_ep_group)
            body['local_ep_group_id'] = _local_epg
        if parsed_args.peer_ep_group:
            _peer_epg = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'endpoint_group', parsed_args.peer_ep_group)
            body['peer_ep_group_id'] = _peer_epg
        if hasattr(parsed_args, 'mtu') and int(parsed_args.mtu) < 68:
            message = _('Invalid MTU value: MTU must be greater than or equal to 68.')
            raise exceptions.CommandError(message)
        if parsed_args.peer_cidrs and parsed_args.local_ep_group:
            message = _('You cannot specify both endpoint groups and peer CIDR(s).')
            raise exceptions.CommandError(message)
        neutronv20.update_dict(parsed_args, body, ['peer_id', 'mtu', 'initiator', 'psk', 'peer_address', 'name', 'description', 'peer_cidrs'])
        return {self.resource: body}