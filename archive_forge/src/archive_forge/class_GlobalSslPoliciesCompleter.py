from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class GlobalSslPoliciesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(GlobalSslPoliciesCompleter, self).__init__(collection='compute.sslPolicies', list_command='compute ssl-policies list --global --uri', **kwargs)