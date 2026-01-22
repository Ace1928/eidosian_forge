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
def IsValidUserPort(val):
    """Validates that a user-provided arg is a valid user port.

  Intended to be used as an argparse validator.

  Args:
    val: str, a string specifying a TCP port number to be validated

  Returns:
    int, the provided port number

  Raises:
    ArgumentTypeError: if the provided port is not an integer outside the
        system port range
  """
    port = int(val)
    if 1024 <= port and port <= 65535:
        return port
    raise argparse.ArgumentTypeError('PORT must be in range [1024, 65535].')