from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.vpn_tunnels import vpn_tunnels_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.external_vpn_gateways import (
from googlecloudsdk.command_lib.compute.routers import flags as router_flags
from googlecloudsdk.command_lib.compute.target_vpn_gateways import (
from googlecloudsdk.command_lib.compute.vpn_gateways import (flags as
from googlecloudsdk.command_lib.compute.vpn_tunnels import flags
def ValidateSimpleSharedSecret(possible_secret):
    """ValidateSimpleSharedSecret checks its argument is a vpn shared secret.

  ValidateSimpleSharedSecret(v) returns v iff v matches [ -~]+.

  Args:
    possible_secret: str, The data to validate as a shared secret.

  Returns:
    The argument, if valid.

  Raises:
    ArgumentTypeError: The argument is not a valid vpn shared secret.
  """
    if not possible_secret:
        raise argparse.ArgumentTypeError('--shared-secret requires a non-empty argument.')
    if re.match(_PRINTABLE_CHARS_PATTERN, possible_secret):
        return possible_secret
    raise argparse.ArgumentTypeError('The argument to --shared-secret is not valid it contains non-printable charcters.')