from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetMaintenancePolicyEnumMapper(messages):
    return arg_utils.ChoiceEnumMapper('--maintenance-policy', messages.NodeGroup.MaintenancePolicyValueValuesEnum, custom_mappings=_MAINTENANCE_POLICY_MAPPINGS)