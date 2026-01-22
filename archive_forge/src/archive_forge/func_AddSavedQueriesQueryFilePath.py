from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddSavedQueriesQueryFilePath(parser, is_required):
    query_file_path_help_text = 'Path to JSON or YAML file that contains the query.'
    parser.add_argument('--query-file-path', required=is_required, help=query_file_path_help_text)