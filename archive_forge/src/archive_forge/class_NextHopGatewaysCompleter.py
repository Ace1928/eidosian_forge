from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class NextHopGatewaysCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(NextHopGatewaysCompleter, self).__init__(collection='compute.nextHopGateways', list_command='alpha compute routes list --uri', **kwargs)