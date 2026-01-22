from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.util.apis import arg_utils
class SubnetworksCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(SubnetworksCompleter, self).__init__(collection='compute.subnetworks', list_command='beta compute networks subnets list --uri', api_version='beta', **kwargs)