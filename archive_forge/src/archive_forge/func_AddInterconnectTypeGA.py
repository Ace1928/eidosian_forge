from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddInterconnectTypeGA(parser):
    """Adds interconnect-type flag to the argparse.ArgumentParser."""
    parser.add_argument('--interconnect-type', choices=_INTERCONNECT_TYPE_CHOICES_GA, required=True, help='      Type of the interconnect.\n      ')