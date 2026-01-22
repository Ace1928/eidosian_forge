import copy
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetSubnet(common.NeutronCommandWithExtraArgs):
    _description = _('Set subnet properties')

    def get_parser(self, prog_name):
        parser = super(SetSubnet, self).get_parser(prog_name)
        parser.add_argument('subnet', metavar='<subnet>', help=_('Subnet to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Updated name of the subnet'))
        dhcp_enable_group = parser.add_mutually_exclusive_group()
        dhcp_enable_group.add_argument('--dhcp', action='store_true', default=None, help=_('Enable DHCP'))
        dhcp_enable_group.add_argument('--no-dhcp', action='store_true', help=_('Disable DHCP'))
        dns_publish_fixed_ip_group = parser.add_mutually_exclusive_group()
        dns_publish_fixed_ip_group.add_argument('--dns-publish-fixed-ip', action='store_true', help=_('Enable publishing fixed IPs in DNS'))
        dns_publish_fixed_ip_group.add_argument('--no-dns-publish-fixed-ip', action='store_true', help=_('Disable publishing fixed IPs in DNS'))
        parser.add_argument('--gateway', metavar='<gateway>', help=_("Specify a gateway for the subnet. The options are: <ip-address>: Specific IP address to use as the gateway, 'none': This subnet will not use a gateway, e.g.: --gateway 192.168.9.1, --gateway none."))
        parser.add_argument('--network-segment', metavar='<network-segment>', help=_('Network segment to associate with this subnet (name or ID). It is only allowed to set the segment if the current value is `None`, the network must also have only one segment and only one subnet can exist on the network.'))
        parser.add_argument('--description', metavar='<description>', help=_('Set subnet description'))
        _tag.add_tag_option_to_parser_for_set(parser, _('subnet'))
        _get_common_parse_arguments(parser, is_create=False)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_subnet(parsed_args.subnet, ignore_missing=False)
        attrs = _get_attrs(self.app.client_manager, parsed_args, is_create=False)
        if 'dns_nameservers' in attrs:
            if not parsed_args.no_dns_nameservers:
                attrs['dns_nameservers'] += obj.dns_nameservers
        elif parsed_args.no_dns_nameservers:
            attrs['dns_nameservers'] = []
        if 'host_routes' in attrs:
            if not parsed_args.no_host_route:
                attrs['host_routes'] += obj.host_routes
        elif parsed_args.no_host_route:
            attrs['host_routes'] = []
        if 'allocation_pools' in attrs:
            if not parsed_args.no_allocation_pool:
                attrs['allocation_pools'] += obj.allocation_pools
        elif parsed_args.no_allocation_pool:
            attrs['allocation_pools'] = []
        if 'service_types' in attrs:
            attrs['service_types'] += obj.service_types
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_subnet(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)
        return