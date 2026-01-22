from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def RoutingPolicyPrimaryDataArg(routing_policy_primary_data):
    """Converts --routing-policy-primary-data flag value to a list of policy data items.

    Args:
      routing_policy_primary_data: String value specified in the
        --routing-policy-primary-data flag.

    Returns:
      A list of forwarding configs in the following format:

    [ 'config1@region1', 'config2@region2',
    'config3' ]
    """
    return routing_policy_primary_data.split(',')