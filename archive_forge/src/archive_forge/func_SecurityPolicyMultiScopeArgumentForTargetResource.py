from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def SecurityPolicyMultiScopeArgumentForTargetResource(resource, required=False, region_hidden=False, scope_flags_usage=compute_flags.ScopeFlagsUsage.GENERATE_DEDICATED_SCOPE_FLAGS, short_help_text=None):
    return compute_flags.ResourceArgument(resource_name='security policy', name='--security-policy', completer=SecurityPoliciesCompleter, plural=False, required=required, global_collection='compute.securityPolicies', regional_collection='compute.regionSecurityPolicies', region_hidden=region_hidden, short_help=(short_help_text or 'The security policy that will be set for this {0}. To remove the policy from this {0} set the policy to an empty string.').format(resource), scope_flags_usage=scope_flags_usage)