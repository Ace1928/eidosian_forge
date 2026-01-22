from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def OrgSecurityPolicyArgument(required=False, plural=False, operation=None):
    return compute_flags.ResourceArgument(name='SECURITY_POLICY', resource_name='security policy', completer=OrgSecurityPoliciesCompleter, plural=plural, required=required, custom_plural='security policies', short_help='Display name or ID of the security policy to {0}.'.format(operation), global_collection='compute.organizationSecurityPolicies')