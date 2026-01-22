from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddConfigFile(parser, hidden=False):
    """Adds config flag."""
    parser.add_argument('--file', hidden=hidden, required=True, help='Path to yaml file containing Delivery Pipeline(s), Target(s) declarative definitions.')