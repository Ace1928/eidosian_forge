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
def ParseX509Parameters(args, is_ca_command):
    """Parses the X509 parameters flags into an API X509Parameters.

  Args:
    args: The parsed argument values.
    is_ca_command: Whether the current command is on a CA. If so, certSign and
      crlSign key usages are added.

  Returns:
    An X509Parameters object.
  """
    preset_profile_set = args.IsKnownAndSpecified('use_preset_profile')
    inline_args = ['key_usages', 'extended_key_usages', 'max_chain_length', 'is_ca_cert', 'unconstrained_chain_length'] + list(_NAME_CONSTRAINT_MAPPINGS.keys())
    has_inline_values = any([args.IsKnownAndSpecified(flag) for flag in inline_args])
    if preset_profile_set and has_inline_values:
        raise exceptions.InvalidArgumentException('--use-preset-profile', '--use-preset-profile may not be specified if one or more of --key-usages, --extended-key-usages, --unconstrained_chain_length or --max-chain-length are specified.')
    if preset_profile_set:
        return preset_profiles.GetPresetX509Parameters(args.use_preset_profile)
    if args.unconstrained_chain_length:
        args.max_chain_length = None
    base_key_usages = args.key_usages or []
    is_ca = is_ca_command or (args.IsKnownAndSpecified('is_ca_cert') and args.is_ca_cert)
    if is_ca:
        base_key_usages.extend(['cert_sign', 'crl_sign'])
    key_usage_dict = {}
    for key_usage in base_key_usages:
        key_usage = text_utils.SnakeCaseToCamelCase(key_usage)
        key_usage_dict[key_usage] = True
    extended_key_usage_dict = {}
    for extended_key_usage in args.extended_key_usages or []:
        extended_key_usage = text_utils.SnakeCaseToCamelCase(extended_key_usage)
        extended_key_usage_dict[extended_key_usage] = True
    messages = privateca_base.GetMessagesModule('v1')
    return messages.X509Parameters(keyUsage=messages.KeyUsage(baseKeyUsage=messages_util.DictToMessageWithErrorCheck(key_usage_dict, messages.KeyUsageOptions), extendedKeyUsage=messages_util.DictToMessageWithErrorCheck(extended_key_usage_dict, messages.ExtendedKeyUsageOptions)), caOptions=messages.CaOptions(isCa=is_ca, maxIssuerPathLength=int(args.max_chain_length) if is_ca and args.max_chain_length is not None else None), nameConstraints=ParseNameConstraints(args, messages))