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
def AddPubsubExportConfigFlags(parser, is_update):
    """Adds Pubsub export config flags to parser."""
    current_group = parser
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group(hidden=True)
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-pubsub-export-config', action='store_true', default=None, hidden=True, help_text='If set, clear the Pubsub Export config from the subscription.')
        current_group = mutual_exclusive_group
    pubsub_export_config_group = current_group.add_argument_group(hidden=True, help="Cloud Pub/Sub Export Config Options. The Cloud Pub/Sub service\n      account associated with the enclosing subscription's parent project\n      (i.e., service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com)\n      must have permission to publish to the destination Cloud Pub/Sub topic.")
    pubsub_export_topic = resource_args.CreateTopicResourceArg('to publish messages to.', flag_name='pubsub-export-topic', positional=False, required=True)
    resource_args.AddResourceArgs(pubsub_export_config_group, [pubsub_export_topic])
    pubsub_export_config_group.add_argument('--pubsub-export-topic-region', required=False, help='The Google Cloud region to which messages are published. For example, us-east1. Do not specify more than one region. If the region you specified is different from the region to which messages were published, egress fees are incurred. If the region is not specified, Pub/Sub uses the same region to which the messages were originally published on a best-effort basis.')