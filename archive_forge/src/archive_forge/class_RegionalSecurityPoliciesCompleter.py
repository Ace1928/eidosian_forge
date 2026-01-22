from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class RegionalSecurityPoliciesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionalSecurityPoliciesCompleter, self).__init__(collection='compute.regionSecurityPolicies', list_command='compute security-policies list --filter=region:* --uri', **kwargs)