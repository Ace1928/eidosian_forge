from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddFullManagement(parser):
    """Adds --full-management flag."""
    parser.add_argument('--full-management', action='store_const', const=True, help='Enable full cluster management type. This is a preview feature.')