from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def _GetIpv4CidrMaskSize(ipv4_cidr_block):
    """Returns the size of IPV4 CIDR block mask in bits.

  Args:
    ipv4_cidr_block: str, the IPV4 CIDR block string to check.

  Returns:
    int, the size of the block mask if ipv4_cidr_block is a valid CIDR block
    string, otherwise None.
  """
    network = ipaddress.IPv4Network(ipv4_cidr_block)
    if network is None:
        return None
    return 32 - (network.num_addresses.bit_length() - 1)