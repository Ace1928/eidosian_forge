from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class NetworkEdgeSecurityServicesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(NetworkEdgeSecurityServicesCompleter, self).__init__(collection='compute.networkEdgeSecurityServices', list_command='compute network-edge-security-services list --uri', **kwargs)