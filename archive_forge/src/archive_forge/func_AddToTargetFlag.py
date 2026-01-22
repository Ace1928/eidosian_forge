from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddToTargetFlag(parser, hidden=False):
    """Adds to-target flag."""
    parser.add_argument('--to-target', hidden=hidden, help='Specifies a target to deliver into upon release creation')