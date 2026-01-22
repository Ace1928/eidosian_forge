from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddFieldsDelimiterArgument(parser):
    """Add the 'fields-terminated-by' argument to the parser."""
    parser.add_argument('--fields-terminated-by', help='Specifies the character that splits column values. The value of this argument has to be a character in Hex ASCII Code. For example, "2C" represents a comma. This flag is only available for MySQL and Postgres. If this flag is not provided, a comma character will be used as the default value.')