from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _GetValidatedAutohealingPolicies(self, holder, client, args, igm_resource):
    health_check = managed_instance_groups_utils.GetHealthCheckUri(holder.resources, args)
    auto_healing_policies = managed_instance_groups_utils.ModifyAutohealingPolicies(igm_resource.autoHealingPolicies, client.messages, args, health_check)
    managed_instance_groups_utils.ValidateAutohealingPolicies(auto_healing_policies)
    return auto_healing_policies