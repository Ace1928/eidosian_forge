from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def _AddSubjectAlternativeNameFlags(parser):
    """Adds the Subject Alternative Name (san) flags.

  This will add --ip-san, --email-san, --dns-san, and --uri-san to the parser.

  Args:
    parser: The parser to add the flags to.
  """
    base.Argument('--email-san', help='One or more comma-separated email Subject Alternative Names.', type=arg_parsers.ArgList(element_type=_StripVal), metavar='EMAIL_SAN').AddToParser(parser)
    base.Argument('--ip-san', help='One or more comma-separated IP Subject Alternative Names.', type=arg_parsers.ArgList(element_type=_StripVal), metavar='IP_SAN').AddToParser(parser)
    base.Argument('--dns-san', help='One or more comma-separated DNS Subject Alternative Names.', type=arg_parsers.ArgList(element_type=_StripVal), metavar='DNS_SAN').AddToParser(parser)
    base.Argument('--uri-san', help='One or more comma-separated URI Subject Alternative Names.', type=arg_parsers.ArgList(element_type=_StripVal), metavar='URI_SAN').AddToParser(parser)