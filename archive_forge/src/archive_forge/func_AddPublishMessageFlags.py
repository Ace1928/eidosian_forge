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
def AddPublishMessageFlags(parser, add_deprecated=False):
    """Adds the flags for building a PubSub message to the parser.

  Args:
    parser: The argparse parser.
    add_deprecated: Whether or not to add the deprecated flags.
  """
    message_help_text = '      The body of the message to publish to the given topic name.\n      Information on message formatting and size limits can be found at:\n      https://cloud.google.com/pubsub/docs/publisher#publish'
    if add_deprecated:
        parser.add_argument('message_body', nargs='?', default=None, help=message_help_text, action=actions.DeprecationAction('MESSAGE_BODY', show_message=lambda _: False, warn=DEPRECATION_FORMAT_STR.format('MESSAGE_BODY', '--message')))
    parser.add_argument('--message', help=message_help_text)
    parser.add_argument('--attribute', type=arg_parsers.ArgDict(max_length=MAX_ATTRIBUTES), help='Comma-separated list of attributes. Each ATTRIBUTE has the form name="value". You can specify up to {0} attributes.'.format(MAX_ATTRIBUTES))
    parser.add_argument('--ordering-key', help='The key for ordering delivery to subscribers. All messages with\n          the same ordering key are sent to subscribers in the order that\n          Pub/Sub receives them.')