from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddExperimentalFeaturesFlag(parser):
    """Adds --experimental-features flag."""
    parser.add_argument('--experimental-features', type=arg_parsers.ArgList(), metavar='FEATURE', help='Enable experimental features. It can only be enabled in ALPHA version.')