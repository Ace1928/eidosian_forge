from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddChangeFeedConditionExpressionArgs(parser):
    help_text = 'Condition expression to overwrite the existing one. For more information, see: https://cloud.google.com/asset-inventory/docs/monitoring-asset-changes#feed_with_condition'
    FeedConditionExpressionArgs(parser, help_text)