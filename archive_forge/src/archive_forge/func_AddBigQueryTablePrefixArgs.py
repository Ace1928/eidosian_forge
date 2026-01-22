from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBigQueryTablePrefixArgs(parser):
    parser.add_argument('--bigquery-table-prefix', metavar='BIGQUERY_TABLE_PREFIX', required=True, type=arg_parsers.RegexpValidator('[\\w]+', '--bigquery-table-prefix must be a BigQuery table name consists of letters, numbers and underscores".'), help='The prefix of the BigQuery tables to which the analysis results will be written. A table name consists of letters, numbers and underscores".')