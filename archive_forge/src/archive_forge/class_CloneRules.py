from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.firewall_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.firewall_policies import firewall_policies_utils
from googlecloudsdk.command_lib.compute.firewall_policies import flags
import six
class CloneRules(base.UpdateCommand):
    """Replace the rules of a Compute Engine organization firewall policy with rules from another policy.

  *{command}* is used to replace the rules of organization firewall policies. An
   organization firewall policy is a set of rules that controls access to
   various resources.
  """
    FIREWALL_POLICY_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.FIREWALL_POLICY_ARG = flags.FirewallPolicyArgument(required=True, operation='clone the rules to')
        cls.FIREWALL_POLICY_ARG.AddArgument(parser, operation_type='clone-rules')
        flags.AddArgsCloneRules(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        ref = self.FIREWALL_POLICY_ARG.ResolveAsResource(args, holder.resources, with_project=False)
        org_firewall_policy = client.OrgFirewallPolicy(ref=ref, compute_client=holder.client, resources=holder.resources, version=six.text_type(self.ReleaseTrack()).lower())
        dest_fp_id = firewall_policies_utils.GetFirewallPolicyId(org_firewall_policy, ref.Name(), organization=args.organization)
        return org_firewall_policy.CloneRules(only_generate_request=False, dest_fp_id=dest_fp_id, source_firewall_policy=args.source_firewall_policy)