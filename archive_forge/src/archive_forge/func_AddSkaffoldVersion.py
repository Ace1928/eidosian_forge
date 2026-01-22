from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSkaffoldVersion(parser):
    """Adds skaffold version flag."""
    parser.add_argument('--skaffold-version', help='Version of the Skaffold binary.')