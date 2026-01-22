from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddFeedPubSubTopicArgs(parser, required):
    parser.add_argument('--pubsub-topic', metavar='PUBSUB_TOPIC', required=required, help='Name of the Cloud Pub/Sub topic to publish to, of the form `projects/PROJECT_ID/topics/TOPIC_ID`. You can list existing topics with `gcloud pubsub topics list --format="text(name)"`')