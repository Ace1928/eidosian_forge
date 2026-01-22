from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddOutputPathArgs(parser, required):
    parser.add_argument('--output-path', metavar='OUTPUT_PATH', required=required, type=arg_parsers.RegexpValidator('^gs://.*', '--output-path must be a Google Cloud Storage URI starting with "gs://". For example, "gs://bucket_name/object_name"'), help='Google Cloud Storage URI where the results will go. URI must start with "gs://". For example, "gs://bucket_name/object_name"')