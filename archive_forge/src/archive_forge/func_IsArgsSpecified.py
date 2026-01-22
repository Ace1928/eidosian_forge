from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def IsArgsSpecified(args):
    """Returns true if at least one of the flags for secrets is specified.

  Args:
    args: Argparse namespace.

  Returns:
    True if at least one of the flags for secrets is specified.
  """
    secrets_flags = {'--set-secrets', '--update-secrets', '--remove-secrets', '--clear-secrets'}
    specified_flags = set(args.GetSpecifiedArgNames())
    return bool(specified_flags.intersection(secrets_flags))