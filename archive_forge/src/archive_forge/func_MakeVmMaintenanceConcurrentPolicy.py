from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def MakeVmMaintenanceConcurrentPolicy(policy_ref, args, messages):
    """Creates a VM Maintenance concurrency limit policy message from args."""
    concurrency_control_group = messages.ResourcePolicyVmMaintenancePolicyConcurrencyControl(concurrencyLimit=args.max_percent)
    vm_policy = messages.ResourcePolicyVmMaintenancePolicy(concurrencyControlGroup=concurrency_control_group)
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, vmMaintenancePolicy=vm_policy)