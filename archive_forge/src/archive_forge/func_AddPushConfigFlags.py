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
def AddPushConfigFlags(parser, required=False, is_update=False):
    """Adds flags for push subscriptions to the parser."""
    parser.add_argument('--push-endpoint', required=required, help='A URL to use as the endpoint for this subscription. This will also automatically set the subscription type to PUSH.')
    parser.add_argument('--push-auth-service-account', required=False, dest='SERVICE_ACCOUNT_EMAIL', help='Service account email used as the identity for the generated Open ID Connect token for authenticated push.')
    parser.add_argument('--push-auth-token-audience', required=False, dest='OPTIONAL_AUDIENCE_OVERRIDE', help='Audience used in the generated Open ID Connect token for authenticated push. If not specified, it will be set to the push-endpoint.')
    current_group = parser
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-push-no-wrapper-config', action='store_true', help_text='If set, clear the NoWrapper config from the subscription.')
        current_group = mutual_exclusive_group
    definition_group = current_group.add_group(mutex=False, help='NoWrapper Config Options.', required=False)
    AddBooleanFlag(parser=definition_group, flag_name='push-no-wrapper', help_text='When set, the message data is delivered directly as the HTTP body.', action='store_true', required=True)
    AddBooleanFlag(parser=definition_group, flag_name='push-no-wrapper-write-metadata', help_text='When true, writes the Pub/Sub message metadata to `x-goog-pubsub-<KEY>:<VAL>` headers of the HTTP request. Writes the Pub/Sub message attributes to `<KEY>:<VAL>` headers of the HTTP request.', action='store_true', required=False)