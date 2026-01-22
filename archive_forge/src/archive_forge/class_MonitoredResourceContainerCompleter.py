from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
class MonitoredResourceContainerCompleter(completers.ResourceParamCompleter):
    """The monitored resource container completer."""

    def __init__(self, **kwargs):
        super(MonitoredResourceContainerCompleter, self).__init__(collection='monitoring.locations.global.metricsScopes', param='monitoredResourceContainerName', **kwargs)