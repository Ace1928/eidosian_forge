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
def AddNameConstraintParameterFlags(parser):
    """Adds flags for inline name constraint x509 parameters.

  Args:
    parser: The parser to add the flags to.
  """
    base.Argument('--name-constraints-critical', help='Indicates whether or not name constraints are marked as critical. Name constraints are considered critical unless explicitly set to false.', default=True, action='store_true').AddToParser(parser)
    base.Argument('--name-permitted-dns', help='One or more comma-separated  DNS names which are permitted to be issued certificates. Any DNS name that can be constructed by simply adding zero or more labels to the left-hand side of the name satisfies the name constraint. For example, `example.com`, `www.example.com`, `www.sub.example.com` would satisfy `example.com`, while `example1.com` does not.', metavar='NAME_PERMITTED_DNS', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-excluded-dns', metavar='NAME_EXCLUDED_DNS', help='One or more comma-separated DNS names which are excluded from being issued certificates. Any DNS name that can be constructed by simply adding zero or more labels to the left-hand side of the name satisfies the name constraint. For example, `example.com`, `www.example.com`, `www.sub.example.com` would satisfy `example.com`, while `example1.com` does not.', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-permitted-ip', metavar='NAME_PERMITTED_IP', help='One or more comma-separated IP ranges which are permitted to be issued certificates. For IPv4 addresses, the ranges are expressed using CIDR notation as specified in RFC 4632. For IPv6 addresses, the ranges are expressed in similar encoding as IPv4', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-excluded-ip', metavar='NAME_EXCLUDED_IP', help='One or more comma-separated IP ranges which are excluded from being issued certificates. For IPv4 addresses, the ranges are expressed using CIDR notation as specified in RFC 4632. For IPv6 addresses, the ranges are expressed in similar encoding as IPv4', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-permitted-email', metavar='NAME_PERMITTED_EMAIL', help='One or more comma-separated email addresses which are permitted to be issued certificates. The value can be a particular email address, a hostname to indicate all email addresses on that host or a domain with a leading period (e.g. `.example.com`) to indicate all email addresses in that domain.', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-excluded-email', metavar='NAME_EXCLUDED_EMAIL', help='One or more comma-separated emails which are excluded from being issued certificates. The value can be a particular email address, a hostname to indicate all email addresses on that host or a domain with a leading period (e.g. `.example.com`) to indicate all email addresses in that domain.', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-permitted-uri', help='One or more comma-separated URIs which are permitted to be issued certificates. The value can be a hostname or a domain with a leading period (like `.example.com`)', metavar='NAME_PERMITTED_URI', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)
    base.Argument('--name-excluded-uri', metavar='NAME_EXCLUDED_URI', help='One or more comma-separated URIs which are excluded from being issued certificates. The value can be a hostname or a domain with a leading period (like `.example.com`)', type=arg_parsers.ArgList(element_type=_StripVal)).AddToParser(parser)