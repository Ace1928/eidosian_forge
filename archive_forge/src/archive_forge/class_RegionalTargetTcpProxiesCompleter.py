from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class RegionalTargetTcpProxiesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionalTargetTcpProxiesCompleter, self).__init__(collection='compute.regionTargetTcpProxies', list_command='compute target-tcp-proxies list --filter=region:* --uri', **kwargs)