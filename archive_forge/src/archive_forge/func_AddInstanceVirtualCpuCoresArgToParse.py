from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddInstanceVirtualCpuCoresArgToParse(parser, required=True):
    """Adds virtual CPU Cores argument for Instance."""
    parser.add_argument('--virtual-cpu-cores', help='Processor for the instance', type=float, required=required)