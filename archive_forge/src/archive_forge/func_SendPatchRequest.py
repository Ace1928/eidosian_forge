from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def SendPatchRequest(self, client, forwarding_rule_ref, replacement):
    """Create forwarding rule patch request."""
    if forwarding_rule_ref.Collection() == 'compute.forwardingRules':
        return client.apitools_client.forwardingRules.Patch(client.messages.ComputeForwardingRulesPatchRequest(project=forwarding_rule_ref.project, region=forwarding_rule_ref.region, forwardingRule=forwarding_rule_ref.Name(), forwardingRuleResource=replacement))
    return client.apitools_client.globalForwardingRules.Patch(client.messages.ComputeGlobalForwardingRulesPatchRequest(project=forwarding_rule_ref.project, forwardingRule=forwarding_rule_ref.Name(), forwardingRuleResource=replacement))