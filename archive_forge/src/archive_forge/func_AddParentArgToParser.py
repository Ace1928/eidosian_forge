from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddParentArgToParser(parser, required=True, message=''):
    """Adds argument for the TagKey or TagValue's parent to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
    required: Boolean, to enforce --parent as a required flag.
    message: String, replacement help text for flag.
  """
    parser.add_argument('--parent', metavar='PARENT', required=required, help=message if message else 'Parent of the resource.')