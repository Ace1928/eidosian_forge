from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
class FirewallsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(FirewallsCompleter, self).__init__(collection='compute.firewalls', list_command='compute firewall-rules list --uri', **kwargs)