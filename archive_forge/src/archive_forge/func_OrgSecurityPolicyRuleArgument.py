from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def OrgSecurityPolicyRuleArgument(required=False, plural=False, operation=None):
    return compute_flags.ResourceArgument(name='priority', resource_name='security policy rule', completer=OrgSecurityPoliciesCompleter, plural=plural, required=required, global_collection='compute.organizationSecurityPolicies', short_help='Priority of the security policy rule to {}.'.format(operation))