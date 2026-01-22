from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetTypeMapperFlag(messages):
    """Helper to get a choice flag from the commitment type enum."""
    return arg_utils.ChoiceEnumMapper('--type', messages.Commitment.TypeValueValuesEnum, help_str='Type of commitment. `memory-optimized` indicates that the commitment is for memory-optimized VMs.', default='general-purpose', include_filter=lambda x: x != 'TYPE_UNSPECIFIED')