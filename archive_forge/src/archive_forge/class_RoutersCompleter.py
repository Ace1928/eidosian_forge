from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
class RoutersCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RoutersCompleter, self).__init__(collection='compute.routers', list_command='compute routers list --uri', **kwargs)