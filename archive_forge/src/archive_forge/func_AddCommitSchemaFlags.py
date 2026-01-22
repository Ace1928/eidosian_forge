from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddCommitSchemaFlags(parser):
    """Adds the flags for the Schema Definition.

  Args:
    parser: The argparse parser
  """
    definition_group = parser.add_group(mutex=True, help='Schema definition', required=True)
    definition_group.add_argument('--definition', type=str, help='The new definition of the schema.')
    definition_group.add_argument('--definition-file', type=arg_parsers.FileContents(), help='File containing the new schema definition.')
    parser.add_argument('--type', type=str, help='The type of the schema.', required=True)