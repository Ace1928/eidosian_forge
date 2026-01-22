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
def AddEscapeArgument(parser):
    """Add the 'escape' argument to the parser."""
    parser.add_argument('--escape', help='Specifies the character that should appear before a data character that needs to be escaped. The value of this argument has to be a character in Hex ASCII Code. For example, "22" represents double quotes. This flag is only available for MySQL and Postgres. If this flag is not provided, double quotes character will be used as the default value.')