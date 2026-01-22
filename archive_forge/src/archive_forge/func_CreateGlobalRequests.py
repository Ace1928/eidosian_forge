from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
def CreateGlobalRequests(self, client, resources, forwarding_rule_ref, args):
    """Create a globally scoped request."""
    target_ref = utils.GetGlobalTarget(resources, args)
    request = client.messages.ComputeGlobalForwardingRulesSetTargetRequest(forwardingRule=forwarding_rule_ref.Name(), project=forwarding_rule_ref.project, targetReference=client.messages.TargetReference(target=target_ref.SelfLink()))
    return [(client.apitools_client.globalForwardingRules, 'SetTarget', request)]