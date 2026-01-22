from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRequestedLinkCount(parser):
    """Adds requestedLinkCount flag to the argparse.ArgumentParser."""
    parser.add_argument('--requested-link-count', required=True, type=int, help='      Target number of physical links in the link bundle.\n      ')