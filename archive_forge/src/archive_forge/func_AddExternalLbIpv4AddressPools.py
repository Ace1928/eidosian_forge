from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddExternalLbIpv4AddressPools(parser):
    parser.add_argument('--external-lb-ipv4-address-pools', type=arg_parsers.ArgList(), metavar='EXTERNAL_LB_IPV4_ADDRESS', help='\n      IPv4 address pools that are used for data plane load balancing of\n      local control plane clusters. Existing pools cannot be updated\n      after cluster creation; only adding new pools is allowed.\n      Each address pool must be specified as one of the following\n      two types of values:\n        1. A IPv4 address range, for example, "10.0.0.1-10.0.0.10". A range that contains a single IP (e.g. "10.0.0.1-10.0.0.1") is allowed.\n        2. A IPv4 CIDR block, for example, "10.0.0.1/24"\n      Use comma when specifying multiple address pools, for example:\n        --external-lb-ipv4-address-pools 10.0.0.1-10.0.0.10,10.0.0.1/24\n      ')