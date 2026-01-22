from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def GetMonitoredResourceContainerNameFlag(verb):
    """Flag for managing a monitored resource container."""
    return base.Argument('monitored_resource_container_name', metavar='MONITORED_RESOURCE_CONTAINER_NAME', completer=completers.MonitoredResourceContainerCompleter, help='Monitored resource container (example - projects/PROJECT_ID) project you want to {0}.'.format(verb))