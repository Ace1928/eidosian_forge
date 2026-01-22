from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRequestedFeatures(parser):
    """Adds requested-features flag to the argparse.ArgumentParser."""
    parser.add_argument('--requested-features', type=arg_parsers.ArgList(choices=_REQUESTED_FEATURES_CHOICES), metavar='FEATURES', help='      List of features requested for this interconnect.\n      ')