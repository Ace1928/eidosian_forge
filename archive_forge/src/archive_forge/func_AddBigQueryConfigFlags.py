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
def AddBigQueryConfigFlags(parser, is_update):
    """Adds BigQuery config flags to parser."""
    current_group = parser
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-bigquery-config', action='store_true', default=None, help_text='If set, clear the BigQuery config from the subscription.')
        current_group = mutual_exclusive_group
    bigquery_config_group = current_group.add_argument_group(help="BigQuery Config Options. The Cloud Pub/Sub service account\n         associated with the enclosing subscription's parent project (i.e.,\n         service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com)\n         must have permission to write to this BigQuery table.")
    bigquery_config_group.add_argument('--bigquery-table', required=True, help='A BigQuery table  of the form {project}:{dataset_name}.{table_name} to which to write messages for this subscription.')
    bigquery_schema_config_mutually_exclusive_group = bigquery_config_group.add_mutually_exclusive_group()
    AddBooleanFlag(parser=bigquery_schema_config_mutually_exclusive_group, flag_name='use-topic-schema', action='store_true', default=None, help_text="Whether or not to use the schema for the subscription's topic (if it exists) when writing messages to BigQuery. If --drop-unknown-fields is not set, then the BigQuery schema must contain all fields that are present in the topic schema.")
    AddBooleanFlag(parser=bigquery_schema_config_mutually_exclusive_group, flag_name='use-table-schema', action='store_true', default=None, help_text='Whether or not to use the BigQuery table schema when writing messages to BigQuery.')
    AddBooleanFlag(parser=bigquery_config_group, flag_name='write-metadata', action='store_true', default=None, help_text='Whether or not to write message metadata including message ID, publish timestamp, ordering key, and attributes to BigQuery. The subscription name, message_id, and publish_time fields are put in their own columns while all other message properties other than data (for example, an ordering_key, if present) are written to a JSON object in the attributes column.')
    AddBooleanFlag(parser=bigquery_config_group, flag_name='drop-unknown-fields', action='store_true', default=None, help_text='If either --use-topic-schema or --use-table-schema is set, whether or not to ignore fields in the message that do not appear in the BigQuery table schema.')