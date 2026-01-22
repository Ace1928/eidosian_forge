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
def MakeGroupPlacementPolicy(policy_ref, args, messages, track):
    """Creates a Group Placement Resource Policy message from args."""
    availability_domain_count = None
    if args.IsSpecified('availability_domain_count'):
        availability_domain_count = args.availability_domain_count
    collocation = None
    if args.IsSpecified('collocation'):
        collocation = flags.GetCollocationFlagMapper(messages, track).GetEnumForChoice(args.collocation)
    placement_policy = None
    if track == base.ReleaseTrack.ALPHA and args.IsSpecified('scope'):
        scope = flags.GetAvailabilityDomainScopeFlagMapper(messages).GetEnumForChoice(args.scope)
        placement_policy = messages.ResourcePolicyGroupPlacementPolicy(vmCount=args.vm_count, availabilityDomainCount=availability_domain_count, collocation=collocation, scope=scope)
    elif track == base.ReleaseTrack.ALPHA and args.IsSpecified('tpu_topology'):
        placement_policy = messages.ResourcePolicyGroupPlacementPolicy(vmCount=args.vm_count, collocation=collocation, tpuTopology=args.tpu_topology)
    elif track in (base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA) and args.IsSpecified('max_distance'):
        placement_policy = messages.ResourcePolicyGroupPlacementPolicy(vmCount=args.vm_count, collocation=collocation, maxDistance=args.max_distance)
    else:
        placement_policy = messages.ResourcePolicyGroupPlacementPolicy(vmCount=args.vm_count, availabilityDomainCount=availability_domain_count, collocation=collocation)
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, groupPlacementPolicy=placement_policy)