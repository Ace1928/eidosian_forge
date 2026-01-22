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
def AddInlineX509ParametersFlags(parser, is_ca_command, default_max_chain_length=None):
    """Adds flags for providing inline x509 parameters.

  Args:
    parser: The parser to add the flags to.
    is_ca_command: Whether the current command is on a CA. This influences the
      help text, and whether the --is-ca-cert flag is added.
    default_max_chain_length: optional, The default value for maxPathLength to
      use if an explicit value is not specified. If this is omitted or set to
      None, no default max path length will be added.
  """
    resource_name = 'CA' if is_ca_command else 'certificate'
    group = parser.add_group()
    base.Argument('--key-usages', metavar='KEY_USAGES', help='The list of key usages for this {}. This can only be provided if `--use-preset-profile` is not provided.'.format(resource_name), type=arg_parsers.ArgList(element_type=_StripVal, choices=_VALID_KEY_USAGES)).AddToParser(group)
    base.Argument('--extended-key-usages', metavar='EXTENDED_KEY_USAGES', help='The list of extended key usages for this {}. This can only be provided if `--use-preset-profile` is not provided.'.format(resource_name), type=arg_parsers.ArgList(element_type=_StripVal, choices=_VALID_EXTENDED_KEY_USAGES)).AddToParser(group)
    name_constraints_group = group.add_group(help='The x509 name constraints configurations')
    AddNameConstraintParameterFlags(name_constraints_group)
    chain_length_group = group.add_group(mutex=True)
    base.Argument('--max-chain-length', help='Maximum depth of subordinate CAs allowed under this CA for a CA certificate. This can only be provided if neither `--use-preset-profile` nor `--unconstrained-chain-length` are provided.', default=default_max_chain_length).AddToParser(chain_length_group)
    base.Argument('--unconstrained-chain-length', help='If set, allows an unbounded number of subordinate CAs under this newly issued CA certificate. This can only be provided if neither `--use-preset-profile` nor `--max-chain-length` are provided.', action='store_true').AddToParser(chain_length_group)
    if not is_ca_command:
        base.Argument('--is-ca-cert', help='Whether this certificate is for a CertificateAuthority or not. Indicates the Certificate Authority field in the x509 basic constraints extension.', required=False, default=False, action='store_true').AddToParser(group)