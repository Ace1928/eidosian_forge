import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def add_arguments_allowedaddresspairs(self, parser):
    group_aap = parser.add_mutually_exclusive_group()
    group_aap.add_argument('--allowed-address-pair', metavar='ip_address=IP_ADDR|CIDR[,mac_address=MAC_ADDR]', default=[], action='append', dest='allowed_address_pairs', type=utils.str2dict_type(required_keys=['ip_address'], optional_keys=['mac_address']), help=_('Allowed address pair associated with the port. "ip_address" parameter is required. IP address or CIDR can be specified for "ip_address". "mac_address" parameter is optional. You can repeat this option.'))
    group_aap.add_argument('--no-allowed-address-pairs', action='store_true', help=_('Associate no allowed address pairs with the port.'))