from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddJobId(parser, hidden=False):
    """Adds job-id flag."""
    parser.add_argument('--job-id', hidden=hidden, help='Job ID on a rollout resource', required=True)