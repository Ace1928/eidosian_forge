from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util.args import labels_util
def AddDisplayNameArg(parser):
    """Adds the display name arg to the parser."""
    parser.add_argument('--display-name', help='      Human readable name which can optionally be supplied.\n      ')