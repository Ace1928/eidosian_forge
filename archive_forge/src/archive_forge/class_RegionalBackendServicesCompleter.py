from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
class RegionalBackendServicesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionalBackendServicesCompleter, self).__init__(collection='compute.regionBackendServices', list_command='compute backend-services list --filter=region:* --uri', **kwargs)