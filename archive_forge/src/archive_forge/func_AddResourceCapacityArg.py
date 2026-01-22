from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddResourceCapacityArg(parser, resource_name, required=True):
    """Add Capacity arg to arg_parser for a resource called resource_name."""
    parser.add_argument('--capacity', type=arg_parsers.BinarySize(default_unit='GiB', suggested_binary_size_scales=['GiB', 'TiB']), required=required, help='The desired capacity of the {} in GiB or TiB units.If no capacity unit is specified, GiB is assumed.'.format(resource_name))