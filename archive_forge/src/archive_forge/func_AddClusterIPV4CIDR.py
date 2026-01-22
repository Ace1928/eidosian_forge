from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddClusterIPV4CIDR(parser):
    parser.add_argument('--cluster-ipv4-cidr', default='10.0.0.0/17', help='\n      All pods in the cluster are assigned an RFC1918 IPv4 address from this\n      block. This field cannot be changed after creation.\n      ')