from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddAuthorizedNetworks(parser, update=False, hidden=False):
    """Adds the `--authorized-networks` flag."""
    cidr_validator = arg_parsers.RegexpValidator(_CIDR_REGEX, "Must be specified in CIDR notation, also known as 'slash' notation (e.g. 192.168.100.0/24).")
    help_ = "The list of external networks that are allowed to connect to the instance. Specified in CIDR notation, also known as 'slash' notation (e.g. 192.168.100.0/24)."
    if update:
        help_ += '\n\nThe value given for this argument *replaces* the existing list.'
    parser.add_argument('--authorized-networks', type=arg_parsers.ArgList(min_length=1, element_type=cidr_validator), metavar='NETWORK', required=False, default=[], help=help_, hidden=hidden)