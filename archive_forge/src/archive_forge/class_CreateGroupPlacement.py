from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.compute.resource_policies import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateGroupPlacement(base.CreateCommand):
    """Create a Compute Engine group placement resource policy."""

    @staticmethod
    def Args(parser):
        _CommonArgs(parser, compute_api.COMPUTE_ALPHA_API_VERSION, base.ReleaseTrack.ALPHA)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        policy_ref = flags.MakeResourcePolicyArg().ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
        messages = holder.client.messages
        resource_policy = util.MakeGroupPlacementPolicy(policy_ref, args, messages, self.ReleaseTrack())
        create_request = messages.ComputeResourcePoliciesInsertRequest(resourcePolicy=resource_policy, project=policy_ref.project, region=policy_ref.region)
        service = holder.client.apitools_client.resourcePolicies
        return client.MakeRequests([(service, 'Insert', create_request)])[0]