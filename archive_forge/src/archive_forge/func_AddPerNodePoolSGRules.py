from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPerNodePoolSGRules(parser):
    """Adds --disable-per-node-pool-sg-rules flag to parser."""
    parser.add_argument('--disable-per-node-pool-sg-rules', action='store_true', default=None, dest='per_node_pool_sg_rules_disabled', help='Disable the default per node pool subnet security group rules on the control plane security group. When disabled, at least one security group that allows node pools to send traffic to the control plane on ports TCP/443 and TCP/8132 must be provided.')