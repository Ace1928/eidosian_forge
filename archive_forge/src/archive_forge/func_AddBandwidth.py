from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddBandwidth(parser, required):
    """Adds bandwidth flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
    required: A boolean indicates whether the Bandwidth is required.
  """
    help_text = '      Provisioned capacity of the attachment.\n      '
    choices = _BANDWIDTH_CHOICES
    base.ChoiceArgument('--bandwidth', choices=choices, required=required, help_str=help_text).AddToParser(parser)