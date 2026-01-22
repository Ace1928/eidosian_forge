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
def ProcessInstanceTypeAndNodes(args):
    """Ensure that --instance-type and --num-nodes are consistent.

  If --instance-type is DEVELOPMENT, then no --cluster-num-nodes can be
  specified. If --instance-type is PRODUCTION, then --cluster-num-nodes defaults
  to 3 if not specified, but can be any positive value.

  Args:
    args: an argparse namespace.

  Raises:
    exceptions.InvalidArgumentException: If --cluster-num-nodes is specified
        when --instance-type is DEVELOPMENT, or --cluster-num-nodes is not
        positive.

  Returns:
    Number of nodes or None if DEVELOPMENT instance-type.
  """
    msgs = util.GetAdminMessages()
    num_nodes = args.cluster_num_nodes
    instance_type = msgs.Instance.TypeValueValuesEnum(args.instance_type)
    if not args.IsSpecified('cluster_num_nodes'):
        if instance_type == msgs.Instance.TypeValueValuesEnum.PRODUCTION:
            num_nodes = 3
    elif instance_type == msgs.Instance.TypeValueValuesEnum.DEVELOPMENT:
        raise exceptions.InvalidArgumentException('--cluster-num-nodes', 'Cannot set --cluster-num-nodes for DEVELOPMENT instances.')
    elif num_nodes < 1:
        raise exceptions.InvalidArgumentException('--cluster-num-nodes', 'Clusters of PRODUCTION instances must have at least 1 node.')
    return num_nodes