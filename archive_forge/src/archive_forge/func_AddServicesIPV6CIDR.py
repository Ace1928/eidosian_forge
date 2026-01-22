from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddServicesIPV6CIDR(parser):
    parser.add_argument('--services-ipv6-cidr', help='\n      If specified, all services in the cluster are assigned an RFC4193 IPv6\n      address from this block. This field cannot be changed after creation.\n      ')