from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDescriptionArgToParser(parser):
    """Adds argument for the TagKey's or TagValue's description to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('--description', metavar='DESCRIPTION', help='User-assigned description of the TagKey or TagValue. Must not exceed 256 characters.')