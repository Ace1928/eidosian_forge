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
def GetKeyArg(help_text='The DNS key identifier.', is_beta=False):
    return base.Argument('key_id', metavar='KEY-ID', completer=BetaKeyCompleter if is_beta else KeyCompleter, help=help_text)