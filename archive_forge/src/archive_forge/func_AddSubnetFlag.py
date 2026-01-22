from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSubnetFlag(parser):
    """Adds --subnet flag."""
    parser.add_argument('--subnet', help='Specifies the subnet that the VM instances are a part of. `--network=NETWORK` must also be specified, subnet must be a subnetwork of the network specified by the `--network=NETWORK` flag.')