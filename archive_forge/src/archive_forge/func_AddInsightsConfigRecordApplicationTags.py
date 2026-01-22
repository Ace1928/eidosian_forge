from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddInsightsConfigRecordApplicationTags(parser, show_negated_in_help):
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--insights-config-record-application-tags', required=False, help='Allow application tags to be recorded by the query insights\n        feature.', **kwargs)