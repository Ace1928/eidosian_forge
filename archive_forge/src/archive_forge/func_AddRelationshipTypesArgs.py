from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddRelationshipTypesArgs(parser):
    parser.add_argument('--relationship-types', metavar='RELATIONSHIP_TYPES', type=arg_parsers.ArgList(), default=[], help='A list of relationship types (i.e., "INSTANCE_TO_INSTANCEGROUP") to take a snapshot. This argument will only be honoured if content_type=RELATIONSHIP. If specified and non-empty, only relationships matching the specified types will be returned. See http://cloud.google.com/asset-inventory/docs/supported-asset-types for supported relationship types.')