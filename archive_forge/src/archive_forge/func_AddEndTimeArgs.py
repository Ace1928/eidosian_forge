from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddEndTimeArgs(parser):
    parser.add_argument('--end-time', required=False, type=arg_parsers.Datetime.Parse, help='End time of the time window (exclusive) for the asset history. Defaults to current time if not specified. See $ gcloud topic datetimes for information on time formats.')