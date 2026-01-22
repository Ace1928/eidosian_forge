from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
class BetaKeyCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(BetaKeyCompleter, self).__init__(collection='dns.dnsKeys', api_version='v1beta2', list_command='beta dns dns-keys list --format=value(keyTag)', parse_output=True, flags=['zone'], **kwargs)