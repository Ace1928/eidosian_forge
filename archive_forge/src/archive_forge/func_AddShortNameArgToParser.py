from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddShortNameArgToParser(parser):
    """Adds positional argument to parser.

  Args:
    parser: ArgumentInterceptor, an argparse parser.
  """
    parser.add_argument('short_name', metavar='SHORT_NAME', help='User specified, friendly name of the TagKey or TagValue. The field must be 1-63 characters, beginning and ending with an alphanumeric character ([a-z0-9A-Z]) with dashes (-), underscores ( _ ), dots (.), and alphanumerics between. ')