from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def AddScalingArgs(self, required=False, num_nodes_required=False, num_nodes_default=None, add_disable_autoscaling=False, require_all_essential_autoscaling_args=False):
    """Add scaling related arguments."""
    scaling_group = self.parser.add_mutually_exclusive_group(required=required)
    manual_scaling_group = scaling_group.add_group('Manual Scaling')
    manual_scaling_group.add_argument('--num-nodes', help='Number of nodes to serve.', default=num_nodes_default, required=num_nodes_required, type=int, metavar='NUM_NODES')
    if add_disable_autoscaling:
        manual_scaling_group.add_argument('--disable-autoscaling', help='Set this flag and --num-nodes to disable autoscaling. If autoscaling is currently not enabled, setting this flag does nothing.', action='store_true', default=False, required=False, hidden=False)
    autoscaling_group = scaling_group.add_group('Autoscaling', hidden=False)
    autoscaling_group.add_argument('--autoscaling-min-nodes', help='The minimum number of nodes for autoscaling.', default=None, required=require_all_essential_autoscaling_args, type=int, metavar='AUTOSCALING_MIN_NODES')
    autoscaling_group.add_argument('--autoscaling-max-nodes', help='The maximum number of nodes for autoscaling.', default=None, required=require_all_essential_autoscaling_args, type=int, metavar='AUTOSCALING_MAX_NODES')
    autoscaling_group.add_argument('--autoscaling-cpu-target', help='The target CPU utilization percentage for autoscaling. Accepted values are from 10 to 80.', default=None, required=require_all_essential_autoscaling_args, type=int, metavar='AUTOSCALING_CPU_TARGET')
    autoscaling_group.add_argument('--autoscaling-storage-target', help='The target storage utilization gibibytes per node for autoscaling. Accepted values are from 2560 to 5120 for SSD clusters and 8192 to 16384 for HDD clusters.', default=None, required=False, type=int, metavar='AUTOSCALING_STORAGE_TARGET')
    return self