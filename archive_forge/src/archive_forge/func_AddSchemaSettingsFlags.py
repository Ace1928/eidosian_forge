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
def AddSchemaSettingsFlags(parser, is_update=False):
    """Adds the flags for filling the SchemaSettings message.

  Args:
    parser: The argparse parser.
    is_update: (bool) If true, add another group with clear-schema-settings as a
      mutually exclusive argument.
  """
    current_group = parser
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-schema-settings', action='store_true', default=None, help_text='If set, clear the Schema Settings from the topic.')
        current_group = mutual_exclusive_group
    set_schema_settings_group = current_group.add_argument_group(help='Schema settings. The schema that messages published to this topic must conform to and the expected message encoding.')
    schema_help_text = 'that messages published to this topic must conform to.'
    schema = resource_args.CreateSchemaResourceArg(schema_help_text, positional=False, plural=False, required=True)
    resource_args.AddResourceArgs(set_schema_settings_group, [schema])
    set_schema_settings_group.add_argument('--message-encoding', type=arg_parsers.ArgList(element_type=lambda x: str(x).lower(), min_length=1, max_length=1, choices=['json', 'binary']), metavar='ENCODING', help='The encoding of messages validated against the schema.', required=True)
    set_schema_settings_group.add_argument('--first-revision-id', help='The id of the oldest\n      revision allowed for the specified schema.', required=False)
    set_schema_settings_group.add_argument('--last-revision-id', help='The id of the most recent\n      revision allowed for the specified schema', required=False)