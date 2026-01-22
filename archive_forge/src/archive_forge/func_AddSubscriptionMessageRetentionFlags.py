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
def AddSubscriptionMessageRetentionFlags(parser, is_update):
    """Adds flags subscription's message retention properties to the parser."""
    if is_update:
        retention_parser = ParseSubscriptionRetentionDurationWithDefault
        retention_default_help = 'Specify "default" to use the default value.'
    else:
        retention_parser = arg_parsers.Duration()
        retention_default_help = 'The default value is 7 days, the minimum is 10 minutes, and the maximum is 7 days.'
    retention_parser = retention_parser or arg_parsers.Duration()
    AddBooleanFlag(parser=parser, flag_name='retain-acked-messages', action='store_true', default=None, help_text="          Whether or not to retain acknowledged messages. If true,\n          messages are not expunged from the subscription's backlog\n          until they fall out of the --message-retention-duration\n          window. Acknowledged messages are not retained by default. ")
    parser.add_argument('--message-retention-duration', type=retention_parser, help="          How long to retain unacknowledged messages in the\n          subscription's backlog, from the moment a message is\n          published. If --retain-acked-messages is true, this also\n          configures the retention of acknowledged messages. {} {}".format(retention_default_help, DURATION_HELP_STR))