import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
def _get_common_parser(parser):
    parser.add_argument('--name', metavar='<name>', help=_('Name of the firewall rule'))
    parser.add_argument('--description', metavar='<description>', help=_('Description of the firewall rule'))
    parser.add_argument('--protocol', choices=['tcp', 'udp', 'icmp', 'any'], type=nc_utils.convert_to_lowercase, help=_('Protocol for the firewall rule'))
    parser.add_argument('--action', choices=['allow', 'deny', 'reject'], type=nc_utils.convert_to_lowercase, help=_('Action for the firewall rule'))
    parser.add_argument('--ip-version', metavar='<ip-version>', choices=['4', '6'], help=_('Set IP version 4 or 6 (default is 4)'))
    src_ip_group = parser.add_mutually_exclusive_group()
    src_ip_group.add_argument('--source-ip-address', metavar='<source-ip-address>', help=_('Source IP address or subnet'))
    src_ip_group.add_argument('--no-source-ip-address', action='store_true', help=_('Detach source IP address'))
    dst_ip_group = parser.add_mutually_exclusive_group()
    dst_ip_group.add_argument('--destination-ip-address', metavar='<destination-ip-address>', help=_('Destination IP address or subnet'))
    dst_ip_group.add_argument('--no-destination-ip-address', action='store_true', help=_('Detach destination IP address'))
    src_port_group = parser.add_mutually_exclusive_group()
    src_port_group.add_argument('--source-port', metavar='<source-port>', help=_('Source port number or range(integer in [1, 65535] or range like 123:456)'))
    src_port_group.add_argument('--no-source-port', action='store_true', help=_('Detach source port number or range'))
    dst_port_group = parser.add_mutually_exclusive_group()
    dst_port_group.add_argument('--destination-port', metavar='<destination-port>', help=_('Destination port number or range(integer in [1, 65535] or range like 123:456)'))
    dst_port_group.add_argument('--no-destination-port', action='store_true', help=_('Detach destination port number or range'))
    shared_group = parser.add_mutually_exclusive_group()
    shared_group.add_argument('--share', action='store_true', help=_('Share the firewall rule to be used in all projects (by default, it is restricted to be used by the current project).'))
    shared_group.add_argument('--no-share', action='store_true', help=_('Restrict use of the firewall rule to the current project'))
    enable_group = parser.add_mutually_exclusive_group()
    enable_group.add_argument('--enable-rule', action='store_true', help=_('Enable this rule (default is enabled)'))
    enable_group.add_argument('--disable-rule', action='store_true', help=_('Disable this rule'))
    src_fwg_group = parser.add_mutually_exclusive_group()
    src_fwg_group.add_argument('--source-firewall-group', metavar='<source-firewall-group>', help=_('Source firewall group (name or ID)'))
    src_fwg_group.add_argument('--no-source-firewall-group', action='store_true', help=_('No associated destination firewall group'))
    dst_fwg_group = parser.add_mutually_exclusive_group()
    dst_fwg_group.add_argument('--destination-firewall-group', metavar='<destination-firewall-group>', help=_('Destination firewall group (name or ID)'))
    dst_fwg_group.add_argument('--no-destination-firewall-group', action='store_true', help=_('No associated destination firewall group'))
    return parser