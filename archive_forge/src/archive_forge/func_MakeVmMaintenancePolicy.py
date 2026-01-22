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
def MakeVmMaintenancePolicy(policy_ref, args, messages):
    """Creates a VM Maintenance Window Resource Policy message from args."""
    vm_policy = messages.ResourcePolicyVmMaintenancePolicy()
    if args.IsSpecified('daily_cycle'):
        _, daily_cycle, _ = _ParseCycleFrequencyArgs(args, messages)
        vm_policy.maintenanceWindow = messages.ResourcePolicyVmMaintenancePolicyMaintenanceWindow(dailyMaintenanceWindow=daily_cycle)
    elif 1 <= args.concurrency_limit_percent <= 100:
        concurrency_control_group = messages.ResourcePolicyVmMaintenancePolicyConcurrencyControl(concurrencyLimit=args.concurrency_limit_percent)
        vm_policy.concurrencyControlGroup = concurrency_control_group
    else:
        raise ValueError('--concurrency-limit-percent must be greater or equal to 1 and less or equal to 100')
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, vmMaintenancePolicy=vm_policy)