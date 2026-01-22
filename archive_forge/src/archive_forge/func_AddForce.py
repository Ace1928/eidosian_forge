from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddForce(parser, help_text, hidden=False):
    """Adds force flag."""
    parser.add_argument('--force', hidden=hidden, action='store_true', help=help_text)