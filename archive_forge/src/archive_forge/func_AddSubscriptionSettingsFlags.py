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
def AddSubscriptionSettingsFlags(parser, is_update=False, enable_push_to_cps=False):
    """Adds the flags for creating or updating a subscription.

  Args:
    parser: The argparse parser.
    is_update: Whether or not this is for the update operation (vs. create).
    enable_push_to_cps: whether or not to enable Pubsub Export config flags
      support.
  """
    AddAckDeadlineFlag(parser)
    AddPushConfigFlags(parser, is_update=is_update)
    mutex_group = parser.add_mutually_exclusive_group()
    AddBigQueryConfigFlags(mutex_group, is_update)
    AddCloudStorageConfigFlags(mutex_group, is_update)
    if enable_push_to_cps:
        AddPubsubExportConfigFlags(mutex_group, is_update)
    AddSubscriptionMessageRetentionFlags(parser, is_update)
    if not is_update:
        AddBooleanFlag(parser=parser, flag_name='enable-message-ordering', action='store_true', default=None, help_text='Whether to receive messages with the same ordering key in order.\n            If set, messages with the same ordering key are sent to subscribers\n            in the order that Pub/Sub receives them.')
    if not is_update:
        parser.add_argument('--message-filter', type=str, help='Expression to filter messages. If set, Pub/Sub only delivers the\n        messages that match the filter. The expression must be a non-empty\n        string in the [Pub/Sub filtering\n        language](https://cloud.google.com/pubsub/docs/filtering).')
    current_group = parser
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-dead-letter-policy', action='store_true', default=None, help_text='If set, clear the dead letter policy from the subscription.')
        current_group = mutual_exclusive_group
    set_dead_letter_policy_group = current_group.add_argument_group(help="Dead Letter Queue Options. The Cloud Pub/Sub service account\n           associated with the enclosing subscription's parent project (i.e.,\n           service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com)\n           must have permission to Publish() to this topic and Acknowledge()\n           messages on this subscription.")
    dead_letter_topic = resource_args.CreateTopicResourceArg('to publish dead letter messages to.', flag_name='dead-letter-topic', positional=False, required=False)
    resource_args.AddResourceArgs(set_dead_letter_policy_group, [dead_letter_topic])
    set_dead_letter_policy_group.add_argument('--max-delivery-attempts', type=arg_parsers.BoundedInt(5, 100), default=None, help='Maximum number of delivery attempts for any message. The value\n          must be between 5 and 100. Defaults to 5. `--dead-letter-topic`\n          must also be specified.')
    parser.add_argument('--expiration-period', type=ParseExpirationPeriodWithNeverSentinel, help='The subscription will expire if it is inactive for the given\n          period. {} This flag additionally accepts the special value "never" to\n          indicate that the subscription will never expire.'.format(DURATION_HELP_STR))
    current_group = parser
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-retry-policy', action='store_true', default=None, help_text='If set, clear the retry policy from the subscription.')
        current_group = mutual_exclusive_group
    set_retry_policy_group = current_group.add_argument_group(help='Retry Policy Options. Retry policy specifies how Cloud Pub/Sub\n              retries message delivery for this subscription.')
    set_retry_policy_group.add_argument('--min-retry-delay', type=arg_parsers.Duration(lower_bound='0s', upper_bound='600s'), help='The minimum delay between consecutive deliveries of a given\n          message. Value should be between 0 and 600 seconds. Defaults to 10\n          seconds. {}'.format(DURATION_HELP_STR))
    set_retry_policy_group.add_argument('--max-retry-delay', type=arg_parsers.Duration(lower_bound='0s', upper_bound='600s'), help='The maximum delay between consecutive deliveries of a given\n          message. Value should be between 0 and 600 seconds. Defaults to 10\n          seconds. {}'.format(DURATION_HELP_STR))
    help_text_suffix = ''
    if is_update:
        help_text_suffix = ' To disable exactly-once delivery use `--no-enable-exactly-once-delivery`.'
    AddBooleanFlag(parser=parser, flag_name='enable-exactly-once-delivery', action='store_true', default=None, help_text="          Whether or not to enable exactly-once delivery on the subscription.\n          If true, Pub/Sub provides the following guarantees for the delivery\n          of a message with a given value of `message_id` on this\n          subscription: The message sent to a subscriber is guaranteed not to\n          be resent before the message's acknowledgment deadline expires. An\n          acknowledged message will not be resent to a subscriber." + help_text_suffix)